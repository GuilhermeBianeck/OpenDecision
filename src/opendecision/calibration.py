"""Temperature scaling fitted only on a separate calibration split."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .confidence import softmax
from .errors import CalibrationError
from .schemas import PublicModel


class CalibrationProfile(PublicModel):
    """Portable model-bound calibration metadata. Task transfer is not guaranteed."""

    version: int = Field(default=1, ge=1, le=1)
    method: Literal["temperature_scaling"] = "temperature_scaling"
    temperature: float = Field(gt=0, allow_inf_nan=False)
    # Optional per-candidate-count temperatures, keyed by the count as a string.
    # A request whose count is absent falls back to the pooled ``temperature``.
    temperatures: dict[str, float] | None = None
    backend: str
    model: str
    revision: str
    template: str = "default"
    task_family: str = "general"
    max_length: int = 512
    precision: str = "float32"
    dataset_sha256: str | None = None
    split: Literal["calibration"] = "calibration"
    sample_count: int = Field(ge=1)
    # Number of candidates per fitted row. One temperature is shared by every
    # choice count; it was only observed on this range.
    choice_count_min: int | None = Field(default=None, ge=2)
    choice_count_max: int | None = Field(default=None, ge=2)
    nll_before: float = Field(ge=0)
    nll_after: float = Field(ge=0)
    created_at: str

    @model_validator(mode="after")
    def valid_temperatures(self) -> CalibrationProfile:
        for key, value in (self.temperatures or {}).items():
            if not key.isdigit() or int(key) < 2:
                raise ValueError("temperatures keys must be candidate counts of at least two")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("temperatures must be finite and positive")
        return self

    def temperature_for(self, choice_count: int) -> float:
        """The temperature applied to a request with this many candidates."""
        if self.temperatures is not None:
            return self.temperatures.get(str(choice_count), self.temperature)
        return self.temperature

    def transform(self, scores: Sequence[float]) -> list[float]:
        """Apply this profile without changing candidate ordering."""
        return softmax(scores, self.temperature_for(len(scores)))

    def covers(self, choice_count: int) -> bool | None:
        """Whether this many candidates was fitted; None if the profile is silent.

        With per-count temperatures this is exact. With a pooled temperature it
        reports whether the count falls inside the fitted range.
        """
        if self.temperatures is not None:
            return str(choice_count) in self.temperatures
        if self.choice_count_min is None or self.choice_count_max is None:
            return None
        return self.choice_count_min <= choice_count <= self.choice_count_max

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        try:
            return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"Cannot load calibration profile: {exc}") from exc


def _nll(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float | dict[int, float],
) -> float:
    """Mean negative log likelihood under one temperature, or one per candidate count."""
    losses = []
    for row, label in zip(rows, labels):
        value = temperature[len(row)] if isinstance(temperature, dict) else temperature
        peak = max(row)
        shifted = [(item - peak) / value for item in row]
        losses.append(math.log(math.fsum(math.exp(x) for x in shifted)) - shifted[label])
    return math.fsum(losses) / len(losses)


def _minimize(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Minimize NLL over T in [0.05, 100] by golden-section search on inverse temperature."""
    lo, hi = 0.01, 20.0
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(80):
        if _nll(rows, labels, 1 / left) < _nll(rows, labels, 1 / right):
            hi, right = right, left
            left = hi - ratio * (hi - lo)
        else:
            lo, left = left, right
            right = lo + ratio * (hi - lo)
    candidates = [1.0, 0.05, 100.0, 1 / ((lo + hi) / 2)]
    return min(candidates, key=lambda t: _nll(rows, labels, t))


def fit_temperature(
    score_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    backend: str,
    model: str,
    revision: str,
    template: str = "default",
    task_family: str = "general",
    dataset_sha256: str | None = None,
    max_length: int = 512,
    precision: str = "float32",
    per_choice_count: bool = True,
    min_rows_per_count: int = 25,
) -> CalibrationProfile:
    """Minimize calibration NLL over T in [0.05, 100], without SciPy.

    The caller must supply held-out calibration rows, never final test rows.
    Sharpness depends strongly on how many candidates a request has: a scorer
    can be overconfident on yes/no rows and underconfident spread across twelve.
    With ``per_choice_count`` a separate temperature is fitted for every
    candidate count with at least ``min_rows_per_count`` rows, and the pooled
    temperature remains the fallback for counts that were not fitted. Pass
    ``per_choice_count=False`` for a single pooled temperature.

    This is still not per-class or out-of-domain calibration.
    """
    if not score_rows or len(score_rows) != len(labels):
        raise CalibrationError("Provide equally sized, nonempty score rows and labels")
    for row, label in zip(score_rows, labels):
        if len(row) < 2 or not all(math.isfinite(value) for value in row):
            raise CalibrationError("Calibration rows require at least two finite scores")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(row):
            raise CalibrationError("Calibration labels must be valid integer choice indexes")
    if min_rows_per_count < 2:
        raise CalibrationError("min_rows_per_count must be at least two")
    # NLL is convex in inverse temperature. Golden-section search respects that domain.
    temperature = _minimize(score_rows, labels)
    temperatures: dict[str, float] | None = None
    if per_choice_count:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(score_rows):
            grouped[len(row)].append(index)
        fitted = {
            count: _minimize([score_rows[i] for i in indexes], [labels[i] for i in indexes])
            for count, indexes in sorted(grouped.items())
            if len(indexes) >= min_rows_per_count
        }
        temperatures = {str(count): value for count, value in fitted.items()} or None
    effective: float | dict[int, float] = (
        {
            len(row): float((temperatures or {}).get(str(len(row)), temperature))
            for row in score_rows
        }
        if temperatures
        else temperature
    )
    return CalibrationProfile(
        temperature=temperature,
        temperatures=temperatures,
        backend=backend,
        model=model,
        revision=revision,
        template=template,
        task_family=task_family,
        max_length=max_length,
        precision=precision,
        dataset_sha256=dataset_sha256,
        sample_count=len(labels),
        choice_count_min=min(len(row) for row in score_rows),
        choice_count_max=max(len(row) for row in score_rows),
        nll_before=_nll(score_rows, labels, 1.0),
        nll_after=_nll(score_rows, labels, effective),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

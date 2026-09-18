"""Temperature scaling fitted only on a separate calibration split."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from .confidence import softmax
from .errors import CalibrationError
from .schemas import PublicModel


class CalibrationProfile(PublicModel):
    """Portable model-bound calibration metadata. Task transfer is not guaranteed."""

    version: int = Field(default=1, ge=1, le=1)
    method: Literal["temperature_scaling"] = "temperature_scaling"
    temperature: float = Field(gt=0, allow_inf_nan=False)
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

    def transform(self, scores: Sequence[float]) -> list[float]:
        """Apply this profile without changing candidate ordering."""
        return softmax(scores, self.temperature)

    def covers(self, choice_count: int) -> bool | None:
        """Whether this many candidates was seen while fitting; None if unrecorded."""
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


def _nll(rows: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    losses = []
    for row, label in zip(rows, labels):
        peak = max(row)
        shifted = [(value - peak) / temperature for value in row]
        losses.append(math.log(math.fsum(math.exp(x) for x in shifted)) - shifted[label])
    return math.fsum(losses) / len(losses)


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
) -> CalibrationProfile:
    """Minimize calibration NLL over T in [0.05, 100], without SciPy.

    The caller must supply held-out calibration rows, never final test rows.
    This is a global temperature, not per-class or out-of-domain calibration.
    The profile records the range of candidate counts it was fitted on; a
    temperature fitted on binary rows is not evidence for ten-way decisions.
    """
    if not score_rows or len(score_rows) != len(labels):
        raise CalibrationError("Provide equally sized, nonempty score rows and labels")
    for row, label in zip(score_rows, labels):
        if len(row) < 2 or not all(math.isfinite(value) for value in row):
            raise CalibrationError("Calibration rows require at least two finite scores")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(row):
            raise CalibrationError("Calibration labels must be valid integer choice indexes")
    # NLL is convex in inverse temperature. Golden-section search respects that domain.
    lo, hi = 0.01, 20.0
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(80):
        if _nll(score_rows, labels, 1 / left) < _nll(score_rows, labels, 1 / right):
            hi, right = right, left
            left = hi - ratio * (hi - lo)
        else:
            lo, left = left, right
            right = lo + ratio * (hi - lo)
    candidates = [1.0, 0.05, 100.0, 1 / ((lo + hi) / 2)]
    temperature = min(candidates, key=lambda t: _nll(score_rows, labels, t))
    return CalibrationProfile(
        temperature=temperature,
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
        nll_after=_nll(score_rows, labels, temperature),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

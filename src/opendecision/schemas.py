"""Validated public schemas shared by the Python and HTTP interfaces."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DecisionRequest(PublicModel):
    """A finite choice problem. Thresholds apply to the effective distribution."""

    state: str = Field(max_length=262_144)
    question: str = Field(min_length=1, max_length=8192)
    choices: list[str] = Field(min_length=2, max_length=256)
    abstain_threshold: Probability | None = None
    margin_threshold: Probability | None = None
    include_raw_scores: bool = False

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("choices")
    @classmethod
    def valid_choices(cls, value: list[str]) -> list[str]:
        if any(not choice.strip() or len(choice) > 4096 for choice in value):
            raise ValueError("choices must be nonblank and at most 4096 characters")
        if len({choice.strip() for choice in value}) != len(value):
            raise ValueError("choices must be unique, including surrounding whitespace")
        return value


class DecisionResult(PublicModel):
    """Normalized scores are not empirical correctness probabilities.

    ``probabilities`` uses the calibrated distribution when a matching profile
    is supplied; otherwise it equals ``normalized_probabilities``.
    ``confidence`` is always the top-one minus top-two probability margin.
    """

    choice: str | None
    probabilities: dict[str, Probability]
    normalized_probabilities: dict[str, Probability]
    calibrated_probabilities: dict[str, Probability] | None = None
    confidence: Probability
    top_probability: Probability
    abstained: bool
    raw_scores: dict[str, float] | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    backend: str
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_result(self) -> DecisionResult:
        keys = set(self.probabilities)
        if len(keys) < 2:
            raise ValueError("a distribution requires at least two choices")
        for distribution in (
            self.probabilities,
            self.normalized_probabilities,
            self.calibrated_probabilities,
        ):
            if distribution is not None:
                if set(distribution) != keys:
                    raise ValueError("all distributions must use the same choices")
                if not math.isclose(math.fsum(distribution.values()), 1.0, abs_tol=1e-6):
                    raise ValueError("probabilities must sum to one")
        effective = self.calibrated_probabilities or self.normalized_probabilities
        if any(not math.isclose(self.probabilities[k], effective[k], abs_tol=1e-8) for k in keys):
            raise ValueError("probabilities must equal the effective distribution")
        if self.abstained != (self.choice is None):
            raise ValueError("choice must be null exactly when abstained")
        if self.choice is not None and self.choice not in keys:
            raise ValueError("selected choice is absent from the distribution")
        ordered = sorted(self.probabilities.values(), reverse=True)
        if not math.isclose(self.top_probability, ordered[0], abs_tol=1e-8):
            raise ValueError("top_probability must match the distribution")
        if not math.isclose(self.confidence, ordered[0] - ordered[1], abs_tol=1e-8):
            raise ValueError("confidence must equal the top-two margin")
        if self.choice is not None and not math.isclose(
            self.probabilities[self.choice], ordered[0], abs_tol=1e-8
        ):
            raise ValueError("selected choice must maximize probability")
        if self.raw_scores is not None and (
            set(self.raw_scores) != keys or not all(map(math.isfinite, self.raw_scores.values()))
        ):
            raise ValueError("raw scores must be finite and match the choice keys")
        return self


class BooleanResult(PublicModel):
    """``probability`` is the effective yes probability, even if abstained."""

    value: bool | None
    probability: Probability
    decision: DecisionResult


class RankedChoice(PublicModel):
    choice: str
    probability: Probability
    raw_score: float | None = None


class RankingResult(PublicModel):
    ranking: list[RankedChoice]
    decision: DecisionResult

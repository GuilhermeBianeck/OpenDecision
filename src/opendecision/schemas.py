"""Validated public schemas shared by the Python and HTTP interfaces."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

MAX_STATE_CHARACTERS = 262_144

StateValue = str | dict[str, Any] | list[str]
"""Context to decide about: plain text, a record with named fields, or a list of texts."""


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _render_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    raise ValueError("state values must be text, numbers, booleans, null, records, or lists")


def _render_fields(record: dict[str, Any], depth: int) -> list[str]:
    indent = "  " * depth
    lines: list[str] = []
    for key, value in record.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            lines.extend(_render_fields(value, depth + 1))
        elif isinstance(value, list):
            if all(not isinstance(item, (dict, list)) for item in value):
                lines.append(f"{indent}{key}: " + ", ".join(_render_scalar(v) for v in value))
            else:
                lines.append(f"{indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        nested = _render_fields(item, depth + 2)
                        lines.append(
                            f"{indent}  -" + (nested[0][len(indent) + 3 :] if nested else "")
                        )
                        lines.extend(nested[1:])
                    else:
                        lines.append(f"{indent}  - {_render_scalar(item)}")
        else:
            lines.append(f"{indent}{key}: {_render_scalar(value)}")
    return lines


def render_state(state: StateValue) -> str:
    """Deterministic text for a state value, in the caller's field order.

    Text passes through unchanged. A record becomes ``key: value`` lines with
    nested records indented; a list becomes one ``- item`` line per entry.
    Field names are part of what the model reads, so name them meaningfully.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {_render_scalar(item)}" for item in state)
    if isinstance(state, dict):
        return "\n".join(_render_fields(state, 0))
    raise ValueError("state must be text, a record, or a list of texts")


def validate_state(value: StateValue) -> StateValue:
    """Reject unrenderable or oversized state at the boundary, before any model runs."""
    rendered = render_state(value)
    if len(rendered) > MAX_STATE_CHARACTERS:
        raise ValueError(f"state renders to more than {MAX_STATE_CHARACTERS} characters")
    return value


class ChoiceOption(PublicModel):
    """A candidate with optional text that separates it from its neighbours.

    Only ``label`` identifies the option in results. The other fields are shown
    to the model alongside the label; use them when bare labels are ambiguous.
    """

    label: str = Field(min_length=1, max_length=4096)
    description: str | None = Field(default=None, max_length=4096)
    not_for: str | None = Field(default=None, max_length=4096)
    examples: list[Annotated[str, Field(min_length=1, max_length=1024)]] | None = Field(
        default=None, max_length=8
    )

    @field_validator("label")
    @classmethod
    def nonblank_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label must not be blank")
        return value

    @field_validator("examples")
    @classmethod
    def nonblank_examples(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not example.strip() for example in value):
            raise ValueError("examples must not be blank")
        return value

    def render(self) -> str:
        """Text presented to the model for this candidate."""
        text = f"{self.label}: {self.description}" if self.description else self.label
        if self.not_for:
            text += f". Not for: {self.not_for}"
        if self.examples:
            text += ". Examples: " + "; ".join(self.examples)
        return text


def _labels(choices: list[str | ChoiceOption]) -> list[str]:
    return [choice.label if isinstance(choice, ChoiceOption) else choice for choice in choices]


def _validate_choices(value: list[str | ChoiceOption]) -> list[str | ChoiceOption]:
    labels = _labels(value)
    if any(not label.strip() or len(label) > 4096 for label in labels):
        raise ValueError("choices must be nonblank and at most 4096 characters")
    if len({label.strip() for label in labels}) != len(labels):
        raise ValueError("choices must be unique, including surrounding whitespace")
    return value


class DecisionRequest(PublicModel):
    """A finite choice problem. Thresholds apply to the effective distribution.

    ``choices`` may mix bare labels and :class:`ChoiceOption` entries. Results
    are always keyed by label.
    """

    state: StateValue
    question: str = Field(min_length=1, max_length=8192)
    choices: list[str | ChoiceOption] = Field(min_length=2, max_length=256)
    abstain_threshold: Probability | None = None
    margin_threshold: Probability | None = None
    include_raw_scores: bool = False

    _validate_state = field_validator("state")(classmethod(lambda cls, v: validate_state(v)))

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("choices")
    @classmethod
    def valid_choices(cls, value: list[str | ChoiceOption]) -> list[str | ChoiceOption]:
        return _validate_choices(value)

    @property
    def state_text(self) -> str:
        return render_state(self.state)

    @property
    def labels(self) -> list[str]:
        """Result keys, in request order."""
        return _labels(self.choices)

    @property
    def candidate_texts(self) -> list[str]:
        """What the model reads for each candidate, in request order."""
        return [
            choice.render() if isinstance(choice, ChoiceOption) else choice
            for choice in self.choices
        ]


class StatementRequest(PublicModel):
    """A yes/no statement about the state.

    Backends with statement scoring evaluate it directly as entailment versus
    contradiction; others fall back to a two-way choice. ``unsupported_threshold``
    abstains when the state neither supports nor contradicts the statement.
    """

    state: StateValue
    statement: str = Field(min_length=1, max_length=8192)
    abstain_threshold: Probability | None = None
    margin_threshold: Probability | None = None
    unsupported_threshold: Probability | None = None

    _validate_state = field_validator("state")(classmethod(lambda cls, v: validate_state(v)))

    @field_validator("statement")
    @classmethod
    def nonblank_statement(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("statement must not be blank")
        return value

    @property
    def state_text(self) -> str:
        return render_state(self.state)


class ScoreRequest(PublicModel):
    """Rate the state on an ordered rubric.

    ``levels`` are descriptions ordered from the lowest to the highest level.
    Each level is scored as a candidate; the score is the probability-weighted
    level index, so it can fall between two levels.
    """

    state: StateValue
    question: str = Field(min_length=1, max_length=8192)
    levels: list[str] = Field(min_length=2, max_length=10)
    abstain_threshold: Probability | None = None
    margin_threshold: Probability | None = None
    include_raw_scores: bool = False

    _validate_state = field_validator("state")(classmethod(lambda cls, v: validate_state(v)))

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("levels")
    @classmethod
    def valid_levels(cls, value: list[str]) -> list[str]:
        if any(not level.strip() or len(level) > 4096 for level in value):
            raise ValueError("levels must be nonblank descriptions of at most 4096 characters")
        if len({level.strip() for level in value}) != len(value):
            raise ValueError("levels must be unique, including surrounding whitespace")
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
    """``probability`` is the effective yes probability, even if abstained.

    ``method`` is ``"statement"`` when the backend scored the statement directly
    against the state; ``unsupported`` is then the probability that the state
    neither supports nor contradicts it. With ``"binary_choice"`` the statement
    was scored as a two-way choice and ``unsupported`` is null.
    """

    value: bool | None
    probability: Probability
    unsupported: Probability | None = None
    method: Literal["statement", "binary_choice"] = "binary_choice"
    decision: DecisionResult

    @model_validator(mode="after")
    def consistent_method(self) -> BooleanResult:
        if (self.method == "statement") != (self.unsupported is not None):
            raise ValueError("unsupported is reported exactly for statement scoring")
        return self


class ScoreResult(PublicModel):
    """``score`` is the expected level index under the effective distribution.

    ``level`` is the most probable level index, or null when abstained. Keys of
    ``probabilities`` and ``legend`` are level indexes as strings, in rubric order.
    The score is a probability-weighted position, not a calibrated magnitude.
    """

    score: float = Field(ge=0, allow_inf_nan=False)
    level: int | None
    probabilities: dict[str, Probability]
    legend: dict[str, str]
    confidence: Probability
    abstained: bool
    decision: DecisionResult

    @model_validator(mode="after")
    def consistent_score(self) -> ScoreResult:
        indexes = [str(index) for index in range(len(self.legend))]
        if list(self.legend) != indexes or list(self.probabilities) != indexes:
            raise ValueError("levels must be indexed 0..n-1 in rubric order")
        expected = math.fsum(int(k) * p for k, p in self.probabilities.items())
        if not math.isclose(self.score, expected, abs_tol=1e-6):
            raise ValueError("score must be the probability-weighted level index")
        if self.abstained != (self.level is None):
            raise ValueError("level must be null exactly when abstained")
        return self


class RankedChoice(PublicModel):
    choice: str
    probability: Probability
    raw_score: float | None = None


class RankingResult(PublicModel):
    ranking: list[RankedChoice]
    decision: DecisionResult

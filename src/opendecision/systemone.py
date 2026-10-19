"""The shared typed-decision wire format, served on top of :meth:`DecisionModel.ask`.

``POST /v1/systemone`` is the request shape the decision-model ecosystem
converged on: Ollama, llama.cpp, SGLang, vLLM and most open decision models
serve it, and SDKs in many languages send it. A request names a ``state`` and a
map of ``questions``; each question is a ``choice`` among named options, a
``score`` on ordered levels, or a ``noul`` — a yes/no statement answered with
the probability that it is true.

This module only translates. Every question becomes the equivalent OpenDecision
question, all of them are answered by one ``ask`` call over the shared state,
and the answers are translated back, so the shared format inherits the same
backends, calibration and routing as the native API.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import Field, field_validator, model_validator

from opendecision.images import extract_images
from opendecision.schemas import (
    BooleanQuestion,
    BooleanResult,
    ChoiceOption,
    ChoiceQuestion,
    DecisionResult,
    PublicModel,
    ScoreQuestion,
    ScoreResult,
    StateValue,
    render_state,
    validate_state,
)

if TYPE_CHECKING:
    from opendecision.api import DecisionModel


MAX_OPTIONS = 255
MAX_SCORE_LEVELS = 10

# Instructions and criteria may be any JSON value: plain text, a record such as
# {"what": ..., "not_for": ..., "examples": [...]}, or a list.
JsonValue = Union[str, int, float, bool, None, dict[str, Any], list[Any]]


def render(value: Any, depth: int = 0) -> str:
    """Deterministic text for a JSON value, preserving the caller's key order.

    Text passes through. A record becomes ``key: value`` lines with nested
    records indented; a list becomes ``- item`` lines. The model reads the
    field names, so a rubric written as {what, not_for, examples} keeps its
    structure instead of being flattened into one sentence.
    """
    indent = "  " * depth
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{indent}{key}:")
                lines.append(render(item, depth + 1))
            else:
                lines.append(f"{indent}{key}: {render(item, depth + 1).strip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                nested = render(item, depth + 1).splitlines()
                lines.append(f"{indent}- {nested[0].strip()}")
                lines.extend(nested[1:])
            else:
                lines.append(f"{indent}- {render(item, depth + 1).strip()}")
        return "\n".join(lines)
    raise ValueError(f"cannot render a {type(value).__name__} as decision text")


def _text(value: Any, what: str) -> str:
    text = render(value).strip()
    if not text:
        raise ValueError(f"{what} must not be blank")
    return text


class ChoiceSpec(PublicModel):
    """Pick one option. Keys are the option names; values describe them."""

    type: Literal["choice"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] = Field(min_length=2, max_length=MAX_OPTIONS)

    _instructions = field_validator("instructions")(
        classmethod(lambda cls, v: _text(v, "instructions") and v)
    )


class ScoreSpec(PublicModel):
    """Rate on ordered levels, lowest first. Each level may be text or a record."""

    type: Literal["score"]
    instructions: JsonValue
    criteria: list[JsonValue] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)

    _instructions = field_validator("instructions")(
        classmethod(lambda cls, v: _text(v, "instructions") and v)
    )


class NoulCriteria(PublicModel):
    """What counts as true and as false, when the boundary needs spelling out."""

    true: JsonValue = None
    false: JsonValue = None


class NoulSpec(PublicModel):
    """A statement about the state; the answer is the probability it is true."""

    type: Literal["noul"]
    instructions: JsonValue
    criteria: NoulCriteria | None = None

    _instructions = field_validator("instructions")(
        classmethod(lambda cls, v: _text(v, "instructions") and v)
    )


QuestionSpec = Annotated[ChoiceSpec | ScoreSpec | NoulSpec, Field(discriminator="type")]


class SystemOneRequest(PublicModel):
    state: StateValue
    model: str = Field(min_length=1)
    questions: dict[str, QuestionSpec] = Field(min_length=1)
    # Data URIs, a top-level field in the shared format; images found inside
    # ``state`` are moved here and replaced by ``[image N]``.
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def lift_images(cls, data: Any) -> Any:
        if isinstance(data, dict) and "state" in data:
            state, images = extract_images(data["state"], data.get("images"))
            data = {**data, "state": state, "images": images}
        return data

    _validate_state = field_validator("state")(classmethod(lambda cls, v: validate_state(v)))


class ChoiceAnswer(PublicModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(PublicModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(PublicModel):
    type: Literal["noul"] = "noul"
    noul: float


AnswerSpec = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(PublicModel):
    """Tokens read and written. Decisions are read off one forward pass, so
    nothing is generated and ``output_tokens`` is always zero."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)


class SystemOneResponse(PublicModel):
    model: str
    answers: dict[str, AnswerSpec]
    usage: Usage


def _statement(spec: NoulSpec) -> str:
    """The statement, with any true/false definitions folded in as plain clauses."""
    statement = _text(spec.instructions, "instructions")
    if spec.criteria is None:
        return statement
    clauses = []
    if spec.criteria.true is not None:
        clauses.append(f"It counts as true when: {render(spec.criteria.true).strip()}")
    if spec.criteria.false is not None:
        clauses.append(f"It counts as false when: {render(spec.criteria.false).strip()}")
    return "\n".join([statement, *clauses])


def _to_question(spec: ChoiceSpec | ScoreSpec | NoulSpec) -> Any:
    if isinstance(spec, ChoiceSpec):
        return ChoiceQuestion(
            question=_text(spec.instructions, "instructions"),
            choices=[
                ChoiceOption(
                    label=label,
                    description=None if description is None else render(description).strip(),
                )
                for label, description in spec.criteria.items()
            ],
        )
    if isinstance(spec, ScoreSpec):
        return ScoreQuestion(
            question=_text(spec.instructions, "instructions"),
            levels=[_text(level, "score level") for level in spec.criteria],
        )
    return BooleanQuestion(statement=_statement(spec))


def _to_answer(result: Any) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    # The shared format has no abstention, so the argmax is always reported.
    if isinstance(result, ScoreResult):
        return ScoreAnswer(
            score=result.score,
            legend=dict(result.legend),
            probabilities=dict(result.probabilities),
            confidence=result.confidence,
        )
    if isinstance(result, BooleanResult):
        return NoulAnswer(noul=result.probability)
    if isinstance(result, DecisionResult):
        top = max(result.probabilities, key=result.probabilities.get)
        return ChoiceAnswer(
            choice=result.choice if result.choice is not None else top,
            probabilities=dict(result.probabilities),
            confidence=result.confidence,
        )
    raise TypeError(f"unexpected answer type {type(result).__name__}")


def _question_texts(question: Any) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        options = [c.render() if isinstance(c, ChoiceOption) else c for c in question.choices]
        return [question.question, *options]
    if isinstance(question, ScoreQuestion):
        return [question.question, *question.levels]
    return [question.statement]


def _usage(engine: DecisionModel, state: StateValue, questions: dict[str, Any]) -> Usage:
    """Count the state once plus every question's own text.

    Backends with a tokenizer count exactly; ones without (the dependency-free
    demo backend) report whitespace-separated words. Prompt scaffolding the
    backend adds around these texts is not counted, so the figure is the same
    whichever template a checkpoint uses.
    """
    count = getattr(engine.backend, "count_tokens", None) or (lambda text: len(text.split()))
    texts = [render_state(state)]
    for question in questions.values():
        texts.extend(_question_texts(question))
    return Usage(input_tokens=sum(count(text) for text in texts))


def answer(engine: DecisionModel, request: SystemOneRequest) -> SystemOneResponse:
    """Answer every question about the shared state in one ``ask`` call.

    The requested ``model`` name is accepted as given, the way a local runtime
    accepts whatever name an SDK sends by default, and the response reports the
    checkpoint that actually answered.
    """
    questions = {key: _to_question(spec) for key, spec in request.questions.items()}
    results = engine.ask(state=request.state, questions=questions, images=request.images)
    return SystemOneResponse(
        model=engine.backend.model_id,
        answers={key: _to_answer(results[key]) for key in request.questions},
        usage=_usage(engine, request.state, questions),
    )

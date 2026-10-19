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

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, field_validator

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
    validate_state,
)

if TYPE_CHECKING:
    from opendecision.api import DecisionModel


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("instructions must not be blank")
    return value


class ChoiceSpec(PublicModel):
    """Pick one option. Keys are the option names; values describe them."""

    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2)

    _nonblank = field_validator("instructions")(classmethod(lambda cls, v: _nonblank(v)))


class ScoreSpec(PublicModel):
    """Rate on ordered levels, lowest first."""

    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=2, max_length=10)

    _nonblank = field_validator("instructions")(classmethod(lambda cls, v: _nonblank(v)))


class NoulSpec(PublicModel):
    """A statement about the state; the answer is the probability it is true."""

    type: Literal["noul"]
    instructions: str = Field(min_length=1)

    _nonblank = field_validator("instructions")(classmethod(lambda cls, v: _nonblank(v)))


QuestionSpec = Annotated[ChoiceSpec | ScoreSpec | NoulSpec, Field(discriminator="type")]


class SystemOneRequest(PublicModel):
    state: StateValue
    model: str = Field(min_length=1)
    questions: dict[str, QuestionSpec] = Field(min_length=1)

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


class SystemOneResponse(PublicModel):
    model: str
    answers: dict[str, AnswerSpec]


def _to_question(spec: ChoiceSpec | ScoreSpec | NoulSpec) -> Any:
    if isinstance(spec, ChoiceSpec):
        return ChoiceQuestion(
            question=spec.instructions,
            choices=[
                ChoiceOption(label=label, description=description)
                for label, description in spec.criteria.items()
            ],
        )
    if isinstance(spec, ScoreSpec):
        return ScoreQuestion(question=spec.instructions, levels=spec.criteria)
    return BooleanQuestion(statement=spec.instructions)


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


def answer(engine: DecisionModel, request: SystemOneRequest) -> SystemOneResponse:
    """Answer every question about the shared state in one ``ask`` call.

    The requested ``model`` name is accepted as given, the way a local runtime
    accepts whatever name an SDK sends by default, and the response reports the
    checkpoint that actually answered.
    """
    questions = {key: _to_question(spec) for key, spec in request.questions.items()}
    results = engine.ask(state=request.state, questions=questions)
    return SystemOneResponse(
        model=engine.backend.model_id,
        answers={key: _to_answer(results[key]) for key in request.questions},
    )

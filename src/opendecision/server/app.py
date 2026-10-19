"""Local-only by default HTTP service with resident, bounded inference."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from threading import BoundedSemaphore
from typing import Annotated, Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from opendecision import DecisionModel
from opendecision.errors import OpenDecisionError
from opendecision.registry import list_models
from opendecision.schemas import (
    MAX_QUESTIONS_PER_STATE,
    Answer,
    BooleanResult,
    DecisionRequest,
    DecisionResult,
    QuestionsRequest,
    RankingResult,
    ScoreRequest,
    ScoreResult,
    StateValue,
    validate_state,
)
from opendecision.server.body import RequestBody
from opendecision.systemone import SystemOneRequest, SystemOneResponse, answer

MAX_BATCH_SIZE = 64
# The shared decision format allows long states and many options per request.
SYSTEMONE_PATH = "/v1/systemone"
SYSTEMONE_MAX_BODY_BYTES = 33_554_432


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[DecisionRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BooleanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: StateValue
    question: str = Field(min_length=1, max_length=8_192)
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    margin_threshold: float | None = Field(default=None, ge=0, le=1)
    unsupported_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("state")
    @classmethod
    def renderable_state(cls, value: StateValue) -> StateValue:
        return validate_state(value)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class MultiLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: StateValue
    labels: list[Annotated[str, Field(min_length=1, max_length=8_192)]] = Field(
        min_length=1, max_length=MAX_QUESTIONS_PER_STATE
    )
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    margin_threshold: float | None = Field(default=None, ge=0, le=1)
    unsupported_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("state")
    @classmethod
    def renderable_state(cls, value: StateValue) -> StateValue:
        return validate_state(value)

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: list[str]) -> list[str]:
        if any(not label.strip() for label in labels):
            raise ValueError("Labels must not be blank")
        if len(labels) != len(set(labels)):
            raise ValueError("Labels must be unique")
        return labels


def create_app(
    model_name: str = "base",
    device: str = "auto",
    calibration: Any = None,
    *,
    decision_model: Any = None,
    max_concurrency: int = 4,
    batch_size: int = 32,
    max_length: int | None = None,
    template: str = "default",
    precision: str | None = None,
    order_averaging: int = 1,
) -> FastAPI:
    """Create an HTTP service; load its model once when the lifespan starts.

    ``decision_model`` supports embedding and deterministic integration tests.
    The service has no authentication and is intended for loopback binding.
    Saturated inference slots fail with 429 instead of growing an unbounded queue.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    slots = BoundedSemaphore(max_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if decision_model is not None:
            app.state.engine = decision_model
        else:
            app.state.engine = await run_in_threadpool(
                DecisionModel,
                model_name,
                device=device,
                calibration=calibration,
                batch_size=batch_size,
                max_length=max_length,
                template=template,
                precision=precision,
                **({"permutations": order_averaging} if order_averaging != 1 else {}),
            )
            await run_in_threadpool(app.state.engine.backend.ensure_loaded)
        yield
        app.state.engine = None

    app = FastAPI(
        title="OpenDecision local API",
        description=(
            "Resident local decision scoring. Normalized scores are not calibrated "
            "unless a matching calibration profile is loaded. Intended for localhost; "
            "network deployments require their own authentication and TLS."
        ),
        version="0.1.0a1",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBody,
        path_limits={SYSTEMONE_PATH: SYSTEMONE_MAX_BODY_BYTES},
        strict_paths=frozenset({SYSTEMONE_PATH}),
    )

    @contextmanager
    def inference_slot() -> Iterator[Any]:
        if not slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Inference capacity is busy; retry shortly",
                headers={"Retry-After": "1"},
            )
        try:
            yield app.state.engine
        finally:
            slots.release()

    @app.exception_handler(ValueError)
    async def request_semantics_error_handler(_request: Any, error: ValueError) -> JSONResponse:
        # Requests that pass schema validation but the loaded model cannot honour.
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(OpenDecisionError)
    async def decision_error_handler(_request: Any, error: OpenDecisionError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": type(error).__name__, "message": str(error)}},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_name,
            "inference": "local",
            "telemetry": False,
            "supports_statements": bool(getattr(app.state.engine, "supports_statements", False)),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"models": list_models(), "active_model": model_name}

    @app.post("/v1/decide", response_model=DecisionResult)
    def decide(request: DecisionRequest) -> DecisionResult:
        with inference_slot() as engine:
            return engine.choose(**request.model_dump())

    @app.post("/v1/decide/batch", response_model=list[DecisionResult])
    def batch(request: BatchRequest) -> list[DecisionResult]:
        with inference_slot() as engine:
            return engine.choose_batch(request.requests)

    @app.post("/v1/rank", response_model=RankingResult)
    def rank(request: DecisionRequest) -> RankingResult:
        with inference_slot() as engine:
            return engine.rank(**request.model_dump())

    @app.post("/v1/ask", response_model=dict[str, Answer])
    def ask(request: QuestionsRequest) -> dict[str, Answer]:
        with inference_slot() as engine:
            return engine.ask(state=request.state, questions=request.questions)

    @app.post(SYSTEMONE_PATH, response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> SystemOneResponse:
        """The shared typed-decision request served across the decision-model ecosystem."""
        with inference_slot() as engine:
            return answer(engine, request)

    @app.post("/v1/score", response_model=ScoreResult)
    def score(request: ScoreRequest) -> ScoreResult:
        with inference_slot() as engine:
            return engine.score(**request.model_dump())

    @app.post("/v1/boolean", response_model=BooleanResult)
    def boolean(request: BooleanRequest) -> BooleanResult:
        with inference_slot() as engine:
            return engine.boolean(**request.model_dump())

    @app.post("/v1/multi-label", response_model=dict[str, BooleanResult])
    def multi_label(request: MultiLabelRequest) -> dict[str, BooleanResult]:
        with inference_slot() as engine:
            return engine.multi_label(**request.model_dump())

    return app

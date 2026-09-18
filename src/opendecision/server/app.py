"""Local-only by default HTTP service with resident, bounded inference."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from threading import BoundedSemaphore
from typing import Annotated, Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from opendecision import DecisionModel
from opendecision.errors import OpenDecisionError
from opendecision.registry import list_models
from opendecision.schemas import BooleanResult, DecisionRequest, DecisionResult, RankingResult

MAX_BODY_BYTES = 1_048_576
MAX_BATCH_SIZE = 64


class BodySizeLimit:
    """Bound request bytes, including requests without a Content-Length header."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        if length < 0 or length > self.max_bytes:
            await JSONResponse({"detail": "Request body exceeds 1 MiB"}, status_code=413)(
                scope, receive, send
            )
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"detail": "Request body exceeds 1 MiB"}, status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        consumed = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[DecisionRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BooleanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str = Field(max_length=262_144)
    question: str = Field(min_length=1, max_length=8_192)
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    margin_threshold: float | None = Field(default=None, ge=0, le=1)
    unsupported_threshold: float | None = Field(default=None, ge=0, le=1)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class MultiLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str = Field(max_length=262_144)
    labels: list[Annotated[str, Field(min_length=1, max_length=8_192)]] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE
    )
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    margin_threshold: float | None = Field(default=None, ge=0, le=1)
    unsupported_threshold: float | None = Field(default=None, ge=0, le=1)

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
    max_length: int = 512,
    template: str = "default",
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
    app.add_middleware(BodySizeLimit)

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

    @app.post("/v1/boolean", response_model=BooleanResult)
    def boolean(request: BooleanRequest) -> BooleanResult:
        with inference_slot() as engine:
            return engine.boolean(**request.model_dump())

    @app.post("/v1/multi-label", response_model=dict[str, BooleanResult])
    def multi_label(request: MultiLabelRequest) -> dict[str, BooleanResult]:
        with inference_slot() as engine:
            return engine.multi_label(**request.model_dump())

    return app

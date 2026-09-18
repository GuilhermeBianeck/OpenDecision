"""Public local decision API. No inference network calls or prompt logging."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .calibration import CalibrationProfile
from .confidence import margin, softmax
from .errors import BackendError, CalibrationError
from .schemas import (
    BooleanResult,
    DecisionRequest,
    DecisionResult,
    RankedChoice,
    RankingResult,
    ScoreRequest,
    ScoreResult,
    StatementRequest,
)

if TYPE_CHECKING:
    from .backends.base import DecisionBackend


class DecisionModel:
    """Load a cached open-weight scorer, or supply a custom backend.

    Download weights explicitly with ``opendecision pull base`` first.
    The per-instance lock bounds simultaneous model inference; use batches
    to improve throughput. Normalized scores alone are not calibrated.
    """

    def __init__(
        self,
        name: str = "base",
        *,
        device: str = "auto",
        backend: DecisionBackend | None = None,
        calibration: CalibrationProfile | str | Path | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        template: str = "default",
        **backend_options: object,
    ) -> None:
        from .registry import create_backend

        if not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if not 32 <= max_length <= 8192:
            raise ValueError("max_length must be between 32 and 8192")
        started = time.perf_counter()
        self.backend = (
            backend
            if backend is not None
            else create_backend(
                name,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                template=template,
                **backend_options,
            )
        )
        self.initialization_ms = (time.perf_counter() - started) * 1000
        self.template = template
        self.max_length = max_length
        self.batch_size = batch_size
        self.calibration = (
            CalibrationProfile.load(calibration)
            if isinstance(calibration, (str, Path))
            else calibration
        )
        self._lock = threading.Lock()
        if self.calibration is not None:
            expected = {
                "backend": self.backend.name,
                "model": self.backend.model_id,
                "revision": self.backend.revision,
                "template": self.template,
                "max_length": self.max_length,
                "precision": self.backend.precision,
            }
            mismatched = [
                k for k, value in expected.items() if getattr(self.calibration, k) != value
            ]
            if mismatched:
                raise CalibrationError("Calibration identity mismatch: " + ", ".join(mismatched))

    @property
    def load_time_ms(self) -> float | None:
        """Actual checkpoint load time, available after the lazy backend has loaded."""
        return getattr(self.backend, "load_time_ms", None)

    def choose(
        self,
        *,
        state: str,
        question: str,
        choices: list[str],
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
        include_raw_scores: bool = False,
    ) -> DecisionResult:
        """Score choices; abstain below either optional probability or margin threshold."""
        request = DecisionRequest(
            state=state,
            question=question,
            choices=choices,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            include_raw_scores=include_raw_scores,
        )
        return self.choose_batch([request])[0]

    def choose_batch(self, requests: list[DecisionRequest]) -> list[DecisionResult]:
        """Score a request batch. Latency is shared batch wall time, not a percentile sample."""
        validated = [DecisionRequest.model_validate(request) for request in requests]
        if not validated:
            return []
        if len(validated) > 1024:
            raise ValueError("at most 1024 requests per batch")
        started = time.perf_counter()
        with self._lock:
            inference_start = time.perf_counter()
            rows = self.backend.score_batch(validated)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            if len(rows) != len(validated):
                raise BackendError("Backend returned the wrong number of score rows")
            elapsed_ms = (time.perf_counter() - started) * 1000
            return [
                self._result(request, row, elapsed_ms, inference_ms, len(validated))
                for request, row in zip(validated, rows)
            ]

    def _result(
        self,
        request: DecisionRequest,
        scores: list[float],
        elapsed_ms: float,
        inference_ms: float,
        batch_count: int,
        *,
        force_abstain: bool = False,
        extra_metadata: dict[str, object] | None = None,
    ) -> DecisionResult:
        if len(scores) != len(request.choices) or not all(math.isfinite(s) for s in scores):
            raise BackendError("Backend must return exactly one finite score per choice")
        normalized = softmax(scores)
        calibrated = self.calibration.transform(scores) if self.calibration else None
        effective = calibrated if calibrated is not None else normalized
        # Deterministic tie-breaking is independent of caller option ordering.
        top = min(range(len(effective)), key=lambda i: (-effective[i], request.choices[i]))
        confidence = margin(effective)
        abstained = (
            force_abstain
            or (
                request.abstain_threshold is not None and effective[top] < request.abstain_threshold
            )
            or (request.margin_threshold is not None and confidence < request.margin_threshold)
        )
        metadata = {
            "revision": self.backend.revision,
            "device": self.backend.device,
            "precision": self.backend.precision,
            "template": self.template,
            "max_length": self.max_length,
            "calibrated": self.calibration is not None,
            "calibration_covers_choice_count": (
                self.calibration.covers(len(request.choices)) if self.calibration else None
            ),
            "confidence_definition": "top1_minus_top2",
            "batch_size": batch_count,
            "batch_inference_ms": inference_ms,
            "latency_scope": "batch_wall_including_lock",
            "calibration": self.calibration.model_dump() if self.calibration else None,
            **(extra_metadata or {}),
        }
        backend_metadata = getattr(self.backend, "metadata", {})
        if isinstance(backend_metadata, dict):
            metadata["backend_details"] = backend_metadata
        return DecisionResult(
            choice=None if abstained else request.choices[top],
            probabilities=dict(zip(request.choices, effective)),
            normalized_probabilities=dict(zip(request.choices, normalized)),
            calibrated_probabilities=dict(zip(request.choices, calibrated)) if calibrated else None,
            confidence=confidence,
            top_probability=effective[top],
            abstained=abstained,
            raw_scores=dict(zip(request.choices, scores)) if request.include_raw_scores else None,
            latency_ms=elapsed_ms,
            backend=self.backend.name,
            model=self.backend.model_id,
            metadata=metadata,
        )

    @property
    def supports_statements(self) -> bool:
        """True when the backend judges statements as entailed, neutral or contradicted."""
        return bool(getattr(self.backend, "supports_statements", False))

    def boolean(
        self,
        *,
        state: str,
        question: str,
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
        unsupported_threshold: float | None = None,
    ) -> BooleanResult:
        """Return an independent yes/no decision with yes probability.

        ``question`` is read as a statement about the state. With statement
        scoring, ``probability`` is entailment versus contradiction and
        ``unsupported`` is the probability that the state settles neither way;
        ``unsupported_threshold`` abstains above it. Other backends score a
        two-way ``yes``/``no`` choice and cannot honour ``unsupported_threshold``.
        """
        request = StatementRequest(
            state=state,
            statement=question,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            unsupported_threshold=unsupported_threshold,
        )
        return self.statement_batch([request])[0]

    def statement_batch(self, requests: list[StatementRequest]) -> list[BooleanResult]:
        """Score independent statements; each result is its own yes/no distribution."""
        validated = [StatementRequest.model_validate(request) for request in requests]
        if not validated:
            return []
        if len(validated) > 1024:
            raise ValueError("at most 1024 statements per batch")
        if not self.supports_statements:
            if any(r.unsupported_threshold is not None for r in validated):
                raise ValueError(
                    f"unsupported_threshold requires statement scoring; backend "
                    f"{self.backend.name!r} scores statements as a two-way choice"
                )
            choices = self.choose_batch([self._as_choice(request) for request in validated])
            return [self._boolean_result(result) for result in choices]
        started = time.perf_counter()
        with self._lock:
            inference_start = time.perf_counter()
            rows = self.backend.score_statements(validated)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            if len(rows) != len(validated):
                raise BackendError("Backend returned the wrong number of statement rows")
            elapsed_ms = (time.perf_counter() - started) * 1000
            return [
                self._statement_result(request, row, elapsed_ms, inference_ms, len(validated))
                for request, row in zip(validated, rows)
            ]

    @staticmethod
    def _as_choice(request: StatementRequest) -> DecisionRequest:
        return DecisionRequest(
            state=request.state,
            question=request.statement,
            choices=["yes", "no"],
            abstain_threshold=request.abstain_threshold,
            margin_threshold=request.margin_threshold,
        )

    def _statement_result(
        self,
        request: StatementRequest,
        logits: list[float],
        elapsed_ms: float,
        inference_ms: float,
        batch_count: int,
    ) -> BooleanResult:
        if len(logits) != 3 or not all(math.isfinite(value) for value in logits):
            raise BackendError("Statement backends must return three finite logits")
        entailment, neutral, contradiction = logits
        # Yes versus no is decided between support and contradiction; the neutral
        # mass is the separate, absolute signal that the state settles neither.
        unsupported = softmax([entailment, neutral, contradiction])[1]
        force_abstain = (
            request.unsupported_threshold is not None
            and unsupported > request.unsupported_threshold
        )
        decision = self._result(
            self._as_choice(request),
            [entailment, contradiction],
            elapsed_ms,
            inference_ms,
            batch_count,
            force_abstain=force_abstain,
            extra_metadata={
                "method": "statement",
                "unsupported": unsupported,
                "unsupported_definition": "p(neutral) over entailment, neutral, contradiction",
                "unsupported_threshold": request.unsupported_threshold,
            },
        )
        return BooleanResult(
            value=None if decision.abstained else decision.choice == "yes",
            probability=decision.probabilities["yes"],
            unsupported=unsupported,
            method="statement",
            decision=decision,
        )

    @staticmethod
    def _boolean_result(result: DecisionResult) -> BooleanResult:
        return BooleanResult(
            value=None if result.abstained else result.choice == "yes",
            probability=result.probabilities["yes"],
            method="binary_choice",
            decision=result,
        )

    def rank(
        self,
        *,
        state: str,
        question: str,
        choices: list[str],
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
        include_raw_scores: bool = False,
    ) -> RankingResult:
        """Rank candidates. A full ordering is returned even if the winner is abstained."""
        result = self.choose(
            state=state,
            question=question,
            choices=choices,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            include_raw_scores=include_raw_scores,
        )
        ranking = [
            RankedChoice(
                choice=choice,
                probability=probability,
                raw_score=result.raw_scores[choice] if result.raw_scores else None,
            )
            for choice, probability in sorted(
                result.probabilities.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        return RankingResult(ranking=ranking, decision=result)

    def score(
        self,
        *,
        state: str,
        question: str,
        levels: list[str],
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
        include_raw_scores: bool = False,
    ) -> ScoreResult:
        """Rate the state on ordered levels; the score is the expected level index."""
        request = ScoreRequest(
            state=state,
            question=question,
            levels=levels,
            abstain_threshold=abstain_threshold,
            margin_threshold=margin_threshold,
            include_raw_scores=include_raw_scores,
        )
        return self.score_batch([request])[0]

    def score_batch(self, requests: list[ScoreRequest]) -> list[ScoreResult]:
        """Score rubric requests as candidate decisions over their level descriptions."""
        validated = [ScoreRequest.model_validate(request) for request in requests]
        decisions = self.choose_batch(
            [
                DecisionRequest(
                    state=request.state,
                    question=request.question,
                    choices=request.levels,
                    abstain_threshold=request.abstain_threshold,
                    margin_threshold=request.margin_threshold,
                    include_raw_scores=request.include_raw_scores,
                )
                for request in validated
            ]
        )
        return [
            self._score_result(request, decision) for request, decision in zip(validated, decisions)
        ]

    @staticmethod
    def _score_result(request: ScoreRequest, decision: DecisionResult) -> ScoreResult:
        probabilities = {
            str(index): decision.probabilities[level] for index, level in enumerate(request.levels)
        }
        return ScoreResult(
            score=math.fsum(index * p for index, p in enumerate(probabilities.values())),
            level=None if decision.abstained else request.levels.index(decision.choice),
            probabilities=probabilities,
            legend={str(index): level for index, level in enumerate(request.levels)},
            confidence=decision.confidence,
            abstained=decision.abstained,
            decision=decision,
        )

    def multi_label(
        self,
        *,
        state: str,
        labels: list[str],
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
        unsupported_threshold: float | None = None,
    ) -> dict[str, BooleanResult]:
        """Evaluate each label as its own statement, never a shared softmax."""
        if not labels or len(labels) > 128 or len(set(labels)) != len(labels):
            raise ValueError("provide 1–128 unique labels")
        requests = [
            StatementRequest(
                state=state,
                statement=label,
                abstain_threshold=abstain_threshold,
                margin_threshold=margin_threshold,
                unsupported_threshold=unsupported_threshold,
            )
            for label in labels
        ]
        return dict(zip(labels, self.statement_batch(requests)))

    def decide_many(
        self,
        *,
        state: str,
        questions: dict[str, list[str]],
        abstain_threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> dict[str, DecisionResult]:
        """Batch independent natural-language questions over a shared state."""
        if not questions or len(questions) > 128:
            raise ValueError("provide 1–128 questions")
        requests = [
            DecisionRequest(
                state=state,
                question=question,
                choices=choices,
                abstain_threshold=abstain_threshold,
                margin_threshold=margin_threshold,
            )
            for question, choices in questions.items()
        ]
        return dict(zip(questions, self.choose_batch(requests)))

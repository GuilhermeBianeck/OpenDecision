"""Shared interface for raw, uncalibrated candidate scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from opendecision.schemas import DecisionRequest, StatementRequest


@runtime_checkable
class DecisionBackend(Protocol):
    """Score each candidate independently; larger scores are preferable."""

    name: str
    model_id: str
    revision: str
    device: str
    precision: str

    @property
    def metadata(self) -> dict[str, Any]: ...

    def score_choices(self, *, state: str, question: str, choices: list[str]) -> list[float]: ...

    def score_batch(self, requests: list[DecisionRequest]) -> list[list[float]]: ...


@runtime_checkable
class StatementBackend(DecisionBackend, Protocol):
    """Also scores a statement against the state as entailment, neutral, contradiction.

    ``supports_statements`` is checked before use so a backend can expose the
    method while declining at runtime (for example a non-NLI checkpoint).
    Each returned row holds three raw logits in that fixed order.
    """

    supports_statements: bool

    def score_statements(self, requests: list[StatementRequest]) -> list[list[float]]: ...

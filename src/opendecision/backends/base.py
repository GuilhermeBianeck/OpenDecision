"""Shared interface for raw, uncalibrated candidate scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from opendecision.schemas import DecisionRequest


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

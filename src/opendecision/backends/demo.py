"""Dependency-free infrastructure fixture, never a learned model."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opendecision.schemas import DecisionRequest


class DemoBackend:
    """Deterministic token overlap for testing plumbing, with no quality claim."""

    name = "demo"
    model_id = "opendecision/token-overlap-demo"
    revision = "demo-v1"
    device = "cpu"
    # Keep the fixture identity explicit while using the profile precision value
    # expected by the portable calibration serializer.
    precision = "float64"
    load_time_ms = 0.0
    supports_statements = False

    def ensure_loaded(self) -> None:
        """No model artifacts are needed for the infrastructure fixture."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "runtime": "python",
            "infrastructure_only": True,
            "warning": "Token-overlap fixture; not a learned decision model.",
        }

    def score_choices(self, *, state: str, question: str, choices: list[str]) -> list[float]:
        context = set(re.findall(r"\w+", (state + " " + question).casefold()))
        return [
            float(len(context.intersection(re.findall(r"\w+", choice.casefold()))))
            for choice in choices
        ]

    def score_batch(self, requests: list[DecisionRequest]) -> list[list[float]]:
        return [
            self.score_choices(state=r.state_text, question=r.question, choices=r.candidate_texts)
            for r in requests
        ]

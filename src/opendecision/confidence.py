"""Stable normalization and explicitly defined uncertainty measures."""

import math
from collections.abc import Sequence


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize finite candidate scores without asserting calibration."""
    if len(scores) < 2 or not all(math.isfinite(s) for s in scores):
        raise ValueError("at least two finite scores are required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    peak = max(scores)
    weights = [math.exp((score - peak) / temperature) for score in scores]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def margin(probabilities: Sequence[float]) -> float:
    """Top-one minus top-two probability; this is not a correctness estimate."""
    first, second = sorted(probabilities, reverse=True)[:2]
    return first - second

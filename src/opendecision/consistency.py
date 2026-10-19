"""Consistency checks that need no reference labels to be meaningful.

* **Negation consistency.** ``p(X) + p(not X)`` should be one. The gap is
  reported as the mean and worst absolute deviation over paired statements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import StatementRequest


def load_negation_pairs(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def negation_consistency(model: Any, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """How far ``p(statement) + p(negation)`` is from one, over ``pairs``.

    Each pair needs ``state``, ``statement`` and ``negation``; a ``target`` of
    ``"yes"`` or ``"no"`` for the statement adds both-sides accuracy.
    """
    if not pairs:
        return {"count": 0, "mean_abs_gap": None, "max_abs_gap": None}
    requests = []
    for pair in pairs:
        requests.append(StatementRequest(state=pair["state"], statement=pair["statement"]))
        requests.append(StatementRequest(state=pair["state"], statement=pair["negation"]))
    results = model.statement_batch(requests)
    gaps, sums, both_correct = [], [], []
    for index, pair in enumerate(pairs):
        positive = results[2 * index].probability
        negative = results[2 * index + 1].probability
        sums.append(positive + negative)
        gaps.append(abs(positive + negative - 1))
        if pair.get("target") in ("yes", "no"):
            truth = pair["target"] == "yes"
            both_correct.append((positive > 0.5) == truth and (negative > 0.5) != truth)
    ordered = sorted(gaps)
    return {
        "count": len(pairs),
        "mean_abs_gap": sum(gaps) / len(gaps),
        "median_abs_gap": ordered[len(ordered) // 2],
        "max_abs_gap": ordered[-1],
        "mean_sum": sum(sums) / len(sums),
        "both_sides_accuracy": sum(both_correct) / len(both_correct) if both_correct else None,
    }

"""Consistency checks that need no reference labels to be meaningful.

Two properties a decision model should have whatever its accuracy:

* **Negation consistency.** ``p(X) + p(not X)`` should be one. The gap is
  reported as the mean and worst absolute deviation over paired statements.
* **Appended-context robustness.** A remark appended to the state that carries
  no evidence should not change the decision: neither an off-topic aside nor
  an unsupported opinion recommending a different option.
* **Option-order invariance.** Reordering the same options should not change
  the decision. The flip rate is the share of cases whose choice changes under
  any of the tested orders; the total-variation distance measures how far the
  distribution moves even when the choice holds.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .schemas import DecisionRequest, StatementRequest


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


def _orders(n: int, permutations: int, rng: random.Random) -> list[list[int]]:
    """The reversed order plus seeded shuffles, never the identity, deduplicated."""
    identity = list(range(n))
    orders = [identity[::-1]]
    attempts = 0
    while len(orders) < permutations and attempts < 50 * permutations:
        attempts += 1
        order = identity[:]
        rng.shuffle(order)
        if order != identity and order not in orders:
            orders.append(order)
    return orders


def order_invariance(
    model: Any,
    requests: list[DecisionRequest],
    *,
    permutations: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-ask each choice request with its options reordered.

    ``permutations`` orders are tried per request: the reversal, then seeded
    shuffles. Probabilities are compared by option label, so the comparison is
    unaffected by where an option sits.
    """
    if not requests:
        return {"count": 0, "flip_rate": None, "mean_total_variation": None}
    rng = random.Random(seed)
    originals = model.choose_batch(requests)
    variants: list[tuple[int, DecisionRequest]] = []
    for index, request in enumerate(requests):
        for order in _orders(len(request.choices), permutations, rng):
            reordered = [request.choices[i] for i in order]
            variants.append((index, request.model_copy(update={"choices": reordered})))
    reordered_results = model.choose_batch([request for _, request in variants])
    flipped = [False] * len(requests)
    distances = []
    for (index, _), result in zip(variants, reordered_results, strict=True):
        original = originals[index]
        if result.choice != original.choice:
            flipped[index] = True
        distances.append(
            sum(
                abs(original.probabilities[label] - result.probabilities[label])
                for label in original.probabilities
            )
            / 2
        )
    return {
        "count": len(requests),
        "orders_per_case": permutations,
        "flip_rate": sum(flipped) / len(flipped),
        "mean_total_variation": sum(distances) / len(distances),
        "max_total_variation": max(distances),
    }


# Appended to the state; neither carries evidence about the decision.
OFF_TOPIC_REMARK = "Unrelated note: someone in the office said the weather was lovely this weekend."
OPINION_REMARK = "Note from a colleague, who has not looked into it: I would just go with {option}."


def _append(state: Any, remark: str) -> Any:
    if isinstance(state, str):
        return f"{state}\n\n{remark}"
    if isinstance(state, dict):
        return {**state, "note": remark}
    return [*state, remark]


def appended_context_robustness(
    model: Any, requests: list[DecisionRequest], *, seed: int = 0
) -> dict[str, Any]:
    """Re-ask each choice request with a remark appended to its state.

    ``off_topic_flip_rate`` uses an aside unrelated to the question.
    ``opinion_flip_rate`` appends an unsupported recommendation of one of the
    options the model did not choose, picked with ``seed``; ``opinion_follow_rate``
    is how often the new answer is the recommended option.
    """
    if not requests:
        return {"count": 0, "off_topic_flip_rate": None, "opinion_flip_rate": None}
    rng = random.Random(seed)
    originals = model.choose_batch(requests)
    suggested = []
    for request, result in zip(requests, originals, strict=True):
        labels = [label for label in request.labels if label != result.choice]
        suggested.append(rng.choice(labels))
    off_topic = model.choose_batch(
        [
            request.model_copy(update={"state": _append(request.state, OFF_TOPIC_REMARK)})
            for request in requests
        ]
    )
    opinion = model.choose_batch(
        [
            request.model_copy(
                update={"state": _append(request.state, OPINION_REMARK.format(option=label))}
            )
            for request, label in zip(requests, suggested, strict=True)
        ]
    )
    count = len(requests)
    return {
        "count": count,
        "off_topic_flip_rate": sum(
            a.choice != b.choice for a, b in zip(originals, off_topic, strict=True)
        )
        / count,
        "opinion_flip_rate": sum(
            a.choice != b.choice for a, b in zip(originals, opinion, strict=True)
        )
        / count,
        "opinion_follow_rate": sum(
            b.choice == label for b, label in zip(opinion, suggested, strict=True)
        )
        / count,
    }

"""Metrics for finite-choice decisions; no optional ML dependency is required."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    """Linearly interpolated sample percentile, or None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def binary_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Probability a positive outranks a negative, giving ties half credit."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels, strict=True))
    negative_below = 0
    favorable = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        positive_ties = sum(label for _, label in ordered[index:end])
        negative_ties = end - index - positive_ties
        favorable += positive_ties * (negative_below + negative_ties / 2)
        negative_below += negative_ties
        index = end
    return favorable / (positives * negatives)


def classification_metrics(records: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    """Score records with target, probabilities, choice, confidence and abstained.

    Accuracy counts abstentions as unanswered failures. Distribution metrics use
    argmax independently of abstention. Brier is the unscaled multiclass sum.
    AUROC is only computed within identical option vocabularies with all classes
    represented; an aggregate over arbitrary incompatible choice sets is invalid.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    if not records:
        return {"count": 0}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels = sorted({record["target"] for record in records})
    correctness = []
    predicted_correctness = []
    confidences = []
    nll, brier = [], []
    vocabularies: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        probabilities = record["probabilities"]
        target = record["target"]
        if target not in probabilities:
            raise ValueError("Target must be in the probability distribution")
        if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
            raise ValueError("Probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        prediction = max(probabilities, key=probabilities.get)
        answer = record.get("choice")
        confusion[target][answer if answer is not None else "<abstain>"] += 1
        correctness.append(answer == target)
        predicted_correctness.append(prediction == target)
        confidences.append(probabilities[prediction])
        nll.append(-math.log(max(probabilities[target], 1e-15)))
        brier.append(sum((p - float(option == target)) ** 2 for option, p in probabilities.items()))
        vocabularies[tuple(sorted(probabilities))].append(record)
    recalls, f1_scores = [], []
    for label in labels:
        tp = confusion[label].get(label, 0)
        support = sum(confusion[label].values())
        predicted = sum(row.get(label, 0) for row in confusion.values())
        recalls.append(tp / support)
        f1_scores.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    reliability = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            i
            for i, p in enumerate(confidences)
            if lower <= p < upper or (index == bins - 1 and p == 1)
        ]
        accuracy = mean([predicted_correctness[i] for i in members]) if members else None
        confidence = mean([confidences[i] for i in members]) if members else None
        if members:
            ece += len(members) / len(records) * abs(accuracy - confidence)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": accuracy,
                "mean_top_probability": confidence,
            }
        )
    auc_groups = []
    for vocabulary, subset in sorted(vocabularies.items()):
        aucs = [
            binary_auroc(
                [r["target"] == label for r in subset], [r["probabilities"][label] for r in subset]
            )
            for label in vocabulary
        ]
        valid = all(value is not None for value in aucs)
        auc_groups.append(
            {
                "choices": list(vocabulary),
                "count": len(subset),
                "macro_ovr_auroc": mean(aucs) if valid else None,
                "reason": None
                if valid
                else "At least one class has no positive or negative examples.",
            }
        )
    answered = [r for r in records if r.get("choice") is not None]
    thresholds = []
    for threshold in (0.0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95):
        selected = [
            r for r in records if r.get("choice") is not None and r["confidence"] >= threshold
        ]
        thresholds.append(
            {
                "min_margin": threshold,
                "coverage": len(selected) / len(records),
                "selective_accuracy": mean([r["choice"] == r["target"] for r in selected])
                if selected
                else None,
                "answered": len(selected),
            }
        )
    return {
        "count": len(records),
        "accuracy": mean(correctness),
        "argmax_accuracy": mean(predicted_correctness),
        "balanced_accuracy": mean(recalls),
        "macro_f1": mean(f1_scores),
        "confusion_matrix": {k: dict(v) for k, v in sorted(confusion.items())},
        "negative_log_likelihood": mean(nll),
        "brier_score": mean(brier),
        "ece": ece,
        "reliability_bins": reliability,
        "auroc_by_choice_vocabulary": auc_groups,
        "coverage": len(answered) / len(records),
        "selective_accuracy": mean([r["choice"] == r["target"] for r in answered])
        if answered
        else None,
        "coverage_by_margin_threshold": thresholds,
    }


def ranking_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate complete strict reference rankings using NDCG and pair agreement."""
    ndcgs, pairwise = [], []
    for record in records:
        reference = record["target_ranking"]
        predicted = sorted(
            reference, key=lambda choice: record["probabilities"][choice], reverse=True
        )
        relevance = {choice: len(reference) - index - 1 for index, choice in enumerate(reference)}

        def dcg(order: list[str]) -> float:
            return sum(
                (2 ** relevance[choice] - 1) / math.log2(i + 2) for i, choice in enumerate(order)
            )

        ideal = dcg(reference)
        ndcgs.append(dcg(predicted) / ideal if ideal else 1.0)
        pairs = [(a, b) for i, a in enumerate(reference) for b in reference[i + 1 :]]
        # Tied scores receive half credit, rather than winning by option order.
        probs = record["probabilities"]
        pairwise.append(
            mean(
                [
                    1.0 if probs[a] > probs[b] else 0.5 if probs[a] == probs[b] else 0.0
                    for a, b in pairs
                ]
            )
        )
    return {
        "count": len(records),
        "ndcg": mean(ndcgs) if ndcgs else None,
        "pairwise_agreement": mean(pairwise) if pairwise else None,
    }

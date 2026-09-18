import math

import pytest

from opendecision.metrics import binary_auroc, classification_metrics, percentile, ranking_metrics


def record(target, probabilities, choice=None, abstained=False):
    ordered = sorted(probabilities.values(), reverse=True)
    return {
        "target": target,
        "probabilities": probabilities,
        "choice": None if abstained else choice or max(probabilities, key=probabilities.get),
        "confidence": ordered[0] - ordered[1],
        "abstained": abstained,
    }


def test_hand_calculated_probability_metrics():
    data = [record("a", {"a": 0.8, "b": 0.2}), record("b", {"a": 0.4, "b": 0.6})]
    metrics = classification_metrics(data)
    assert metrics["accuracy"] == 1
    assert metrics["balanced_accuracy"] == 1
    assert metrics["macro_f1"] == 1
    assert metrics["brier_score"] == pytest.approx(0.2)
    assert metrics["negative_log_likelihood"] == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)
    assert metrics["ece"] == pytest.approx(0.3)
    assert metrics["auroc_by_choice_vocabulary"][0]["macro_ovr_auroc"] == 1


def test_abstention_is_not_counted_correct():
    metrics = classification_metrics([record("a", {"a": 0.8, "b": 0.2}, abstained=True)])
    assert metrics["accuracy"] == 0
    assert metrics["argmax_accuracy"] == 1
    assert metrics["coverage"] == 0
    assert metrics["selective_accuracy"] is None
    assert metrics["confusion_matrix"] == {"a": {"<abstain>": 1}}
    assert metrics["auroc_by_choice_vocabulary"][0]["macro_ovr_auroc"] is None


def test_auc_ties_and_undefined_class():
    assert binary_auroc([True, False], [0.5, 0.5]) == 0.5
    assert binary_auroc([False, True], [0.9, 0.1]) == 0
    assert binary_auroc([True, True], [0.9, 0.1]) is None
    assert binary_auroc([True, False, True, False], [0.7, 0.7, 0.9, 0.2]) == 0.875


def test_auroc_does_not_merge_unrelated_vocabularies():
    data = [record("a", {"a": 0.9, "b": 0.1}), record("d", {"c": 0.1, "d": 0.9})]
    assert all(
        group["macro_ovr_auroc"] is None
        for group in classification_metrics(data)["auroc_by_choice_vocabulary"]
    )


def test_invalid_distribution_fails_and_empty_is_explicit():
    with pytest.raises(ValueError, match="sum"):
        classification_metrics([record("a", {"a": 0.8, "b": 0.8})])
    assert classification_metrics([]) == {"count": 0}
    assert percentile([], 0.5) is None
    assert percentile([0, 10], 0.95) == 9.5


def test_ranking_and_tie_pair_agreement():
    perfect = {"target_ranking": ["a", "b", "c"], "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}}
    assert ranking_metrics([perfect])["ndcg"] == 1
    assert ranking_metrics([perfect])["pairwise_agreement"] == 1
    tied = {"target_ranking": ["a", "b"], "probabilities": {"a": 0.5, "b": 0.5}}
    assert ranking_metrics([tied])["pairwise_agreement"] == 0.5

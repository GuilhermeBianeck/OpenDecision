"""Negation consistency and option-order invariance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_core import FixedBackend, StatementBackend

from opendecision import DecisionModel
from opendecision.consistency import (
    load_negation_pairs,
    negation_consistency,
    order_invariance,
)
from opendecision.schemas import DecisionRequest

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "benchmarks" / "datasets" / "negation_pairs.jsonl"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_negation_pairs", ROOT / "scripts" / "generate_negation_pairs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("statement", "negation"),
    [
        (
            "The invoice was issued by Harbor Logistics.",
            "The invoice was not issued by Harbor Logistics.",
        ),
        ("The invoice has been paid.", "The invoice has not been paid."),
        ("The ticket is still open.", "The ticket is no longer open."),
        ("The ticket has high severity.", "The ticket does not have high severity."),
        ("The parcel weighs 12 kg.", "The parcel does not weigh 12 kg."),
        (
            "Mara Lindqvist belongs to the platform team.",
            "Mara Lindqvist does not belong to the platform team.",
        ),
        ("The parcel has not been delivered yet.", None),
    ],
)
def test_negation_rules(statement, negation):
    assert _generator().negate(statement) == negation


def test_committed_pairs_are_well_formed():
    pairs = load_negation_pairs(PAIRS)
    assert len(pairs) >= 90
    assert len({pair["id"] for pair in pairs}) == len(pairs)
    for pair in pairs:
        assert pair["statement"] != pair["negation"]
        assert pair["target"] in ("yes", "no")
        assert any(word in pair["negation"].split() for word in ("not", "no"))


class Fixed(StatementBackend):
    """p(yes) is the same for every statement, so p(X) + p(not X) = 2 p."""


def test_a_constant_model_misses_by_twice_its_offset_from_one_half():
    # Logits [0, -inf-ish, 0] give p(yes) = 0.5 under statement scoring.
    model = DecisionModel(backend=Fixed([0.0, -50.0, 0.0]))
    pairs = [{"state": "s", "statement": "It is so.", "negation": "It is not so.", "target": "yes"}]
    report = negation_consistency(model, pairs)
    assert report["mean_abs_gap"] == pytest.approx(0.0, abs=1e-9)
    model = DecisionModel(backend=Fixed([2.0, -50.0, 0.0]))
    report = negation_consistency(model, pairs)
    p = report["mean_sum"] / 2
    assert p > 0.5
    assert report["mean_abs_gap"] == pytest.approx(2 * p - 1)
    assert report["both_sides_accuracy"] == 0.0  # it says yes to both


def test_empty_inputs_report_none():
    model = DecisionModel("demo")
    assert negation_consistency(model, [])["mean_abs_gap"] is None
    assert order_invariance(model, [])["flip_rate"] is None


REQUESTS = [
    DecisionRequest(
        state="The customer was charged twice for one order.",
        question="Which team should handle this?",
        choices=["billing", "shipping", "technical support", "sales"],
    ),
    DecisionRequest(state="Two choices only.", question="Which?", choices=["left", "right"]),
]


def test_a_content_based_model_is_order_invariant():
    report = order_invariance(DecisionModel("demo"), REQUESTS, permutations=4)
    assert report["flip_rate"] == 0
    assert report["mean_total_variation"] == pytest.approx(0, abs=1e-9)


class Positional(FixedBackend):
    """Always prefers whatever option is listed first."""

    def score_batch(self, requests):
        return [[5.0] + [0.0] * (len(r.choices) - 1) for r in requests]


def test_a_positional_model_flips_every_time():
    report = order_invariance(DecisionModel(backend=Positional()), REQUESTS, permutations=3)
    assert report["flip_rate"] == 1
    # Shuffles that keep the first option in place leave the distribution alone.
    assert report["mean_total_variation"] > 0.5


def test_orders_are_seeded_and_exclude_the_identity():
    module = __import__("opendecision.consistency", fromlist=["_orders"])
    import random

    first = module._orders(5, 4, random.Random(7))
    assert first == module._orders(5, 4, random.Random(7))
    assert list(range(5)) not in first
    assert first[0] == [4, 3, 2, 1, 0]
    assert module._orders(2, 4, random.Random(0)) == [[1, 0]]


def test_cli_writes_a_report(tmp_path):
    from opendecision.cli import main

    output = tmp_path / "consistency.json"
    assert main(["consistency", "--model", "demo", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["negation"]["count"] > 0
    assert report["choice_order"]["count"] > 0


# Measured on the test split's 94 base choice cases with 3 orders each
# (see docs/consistency.md). The bounds sit just above the measurement so a
# regression fails while noise from a re-run does not.
ORDER_FLIP_BOUNDS = {"base": 0.0, "qwen35": 0.25}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ORDER_FLIP_BOUNDS))
def test_option_order_sensitivity_does_not_regress(name, monkeypatch):
    """Set OPENDECISION_TEST_MODELS=base,qwen35 after explicit model pulls."""
    import os

    from opendecision.benchmark import load_dataset

    if name not in os.environ.get("OPENDECISION_TEST_MODELS", "").split(","):
        pytest.skip("Real weights not requested via OPENDECISION_TEST_MODELS")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    requests = [
        case.to_request().model_copy(update={"abstain_threshold": None})
        for case in load_dataset(ROOT / "benchmarks" / "datasets" / "core.jsonl")
        if case.kind == "choice" and case.variant == "base" and case.split == "test"
    ]
    report = order_invariance(DecisionModel(name), requests, permutations=3, seed=0)
    assert report["flip_rate"] <= ORDER_FLIP_BOUNDS[name]

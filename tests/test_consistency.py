"""Negation consistency."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_core import StatementBackend

from opendecision import DecisionModel
from opendecision.consistency import load_negation_pairs, negation_consistency

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


def test_cli_writes_a_report(tmp_path):
    from opendecision.cli import main

    output = tmp_path / "consistency.json"
    assert main(["consistency", "--model", "demo", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["negation"]["count"] > 0

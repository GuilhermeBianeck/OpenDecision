import json
from pathlib import Path

import httpx
import pytest

from opendecision import DecisionModel
from opendecision.benchmark import BenchmarkCase, load_dataset, run_benchmark, write_report
from opendecision.external import (
    ExternalProviderError,
    ExternalUnavailable,
    RemoteDecisionModel,
    create_external_model,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_integrity_and_split_isolation():
    rows = load_dataset(ROOT / "benchmarks/datasets/core.jsonl")
    seeds = load_dataset(ROOT / "benchmarks/datasets/seeds.jsonl")
    assert len(rows) >= 2000
    assert len(seeds) >= 200
    group_splits = {}
    identifiers = {row.id for row in rows}
    for row in rows:
        group_splits.setdefault(row.group_id, set()).add(row.split)
        assert row.base_id is None or row.base_id in identifiers
        assert row.metadata["synthetic"]
        assert row.metadata["human_adjudicated"] is False
        if row.family == "subjective" and row.target:
            assert row.reference_policy
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {row.split for row in rows} == {"train", "calibration", "validation", "test"}


def test_rejects_split_leakage(tmp_path):
    row = load_dataset(ROOT / "benchmarks/datasets/seeds.jsonl")[0].model_dump()
    path = tmp_path / "leaky.jsonl"
    path.write_text(
        json.dumps(row)
        + "\n"
        + json.dumps(
            {
                **row,
                "id": "other",
                "split": "validation" if row["split"] != "validation" else "test",
            }
        )
    )
    with pytest.raises(ValueError, match="Split leakage"):
        load_dataset(path)


def test_subjective_target_requires_policy():
    with pytest.raises(ValueError, match="reference policy"):
        BenchmarkCase(
            id="x",
            family="subjective",
            group_id="x",
            split="test",
            state="A tradeoff",
            question="What action?",
            choices=["a", "b"],
            target="a",
        )


def mixed_cases():
    common = {"split": "test", "variant": "base", "metadata": {"synthetic": True}}
    return [
        BenchmarkCase(
            id="c1",
            family="objective",
            group_id="g1",
            state="billing invoice",
            question="Team?",
            choices=["billing", "technical"],
            target="billing",
            **common,
        ),
        BenchmarkCase(
            id="b1",
            family="verification",
            group_id="g2",
            kind="boolean",
            state="billing invoice",
            question="billing",
            statement="billing",
            choices=["yes", "no"],
            target="yes",
            **common,
        ),
        BenchmarkCase(
            id="b2",
            family="verification",
            group_id="g2",
            kind="boolean",
            state="billing invoice",
            question="shipping",
            statement="shipping delay",
            choices=["yes", "no"],
            target="no",
            **common,
        ),
        BenchmarkCase(
            id="s1",
            family="ordinal",
            group_id="g3",
            kind="score",
            state="outage everywhere",
            question="Severity?",
            levels=["cosmetic", "degraded", "outage everywhere"],
            choices=["cosmetic", "degraded", "outage everywhere"],
            target_level=2,
            target="outage everywhere",
            **common,
        ),
        BenchmarkCase(
            id="r1",
            family="robustness",
            group_id="g4",
            state="not technical, billing",
            question="Team?",
            choices=["billing", "technical"],
            target="billing",
            **common,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "boolean", "choices": ["yes", "no"]},  # missing statement
        {"kind": "boolean", "statement": "s", "choices": ["a", "b"], "target": "a"},
        {"statement": "s"},  # statement on a choice case
        {"kind": "score", "levels": ["a", "b"], "choices": ["a", "b"], "target_level": 5},
        {
            "kind": "score",
            "levels": ["a", "b"],
            "choices": ["a", "b"],
            "target_level": 0,
            "target": "b",
        },
        {"levels": ["a", "b"]},  # levels on a choice case
    ],
)
def test_case_kinds_are_validated(overrides):
    base = {
        "id": "x",
        "family": "objective",
        "group_id": "g",
        "split": "test",
        "state": "s",
        "question": "q",
        "choices": ["a", "b"],
        "target": "a",
    }
    with pytest.raises(ValueError):
        BenchmarkCase(**{**base, **overrides})


def test_mixed_kinds_are_evaluated_with_their_own_primitives(tmp_path):
    report = run_benchmark(DecisionModel("demo"), mixed_cases(), batch_size=2, robustness=True)
    rows = {row["id"]: row for row in report["predictions"]}
    assert rows["c1"]["kind"] == "choice" and rows["c1"]["choice"] == "billing"
    # The demo fixture has no statement scoring and no signal for yes/no words.
    assert rows["b1"]["kind"] == "boolean" and isinstance(rows["b1"]["value"], bool)
    assert rows["b1"]["method"] == "binary_choice"
    assert set(rows["b1"]["probabilities"]) == {"yes", "no"}
    assert rows["s1"]["kind"] == "score" and rows["s1"]["level"] == 2
    assert rows["s1"]["legend"]["2"] == "outage everywhere"
    assert list(rows["s1"]["level_probabilities"]) == ["0", "1", "2"]
    assert report["dataset"]["kinds"] == {"choice": 2, "boolean": 2, "score": 1}
    # Verification and robustness count as objective; the rubric is measured separately.
    assert report["objective"]["count"] == 4
    assert set(report["by_family"]) == {"objective", "verification", "robustness"}
    assert report["by_variant"]["base"]["count"] == 4
    # Every objective row in this fixture is two-way; the rubric is scored separately.
    assert sorted(report["objective_by_choice_count"], key=int) == ["2"]
    assert report["objective_by_choice_count"]["2"]["count"] == 4
    assert report["ordinal"]["count"] == 1 and report["ordinal"]["exact_level_accuracy"] == 1
    assert report["verification"] == {
        "count": 2,
        "statement_scored": 0,
        "mean_unsupported_when_target_no": None,
        "mean_unsupported_when_target_yes": None,
    }
    # Option reversal only probes free choices; repeats cover every kind.
    assert report["robustness"]["choice_order"]["count"] == 2
    assert report["robustness"]["self_consistency"]["count"] == 5
    assert "reversed_choice_order" in rows["c1"] and "reversed_choice_order" not in rows["s1"]
    markdown = Path(write_report(report, tmp_path / "mixed")["markdown"]).read_text()
    assert "Ordinal rubrics: 1 cases" in markdown and "| verification |" in markdown
    assert "## Objective accuracy by candidate count" in markdown


def test_statement_backends_feed_verification_metrics():
    from test_core import StatementBackend

    class SizedStatementBackend(StatementBackend):
        def score_batch(self, requests):
            self.requests.extend(requests)
            return [[float(len(r.choices) - i) for i in range(len(r.choices))] for r in requests]

    backend = SizedStatementBackend([1.0, 2.0, 0.0])
    report = run_benchmark(DecisionModel(backend=backend), mixed_cases())
    rows = {row["id"]: row for row in report["predictions"]}
    assert rows["b1"]["method"] == "statement" and 0 < rows["b1"]["unsupported"] < 1
    assert report["verification"]["statement_scored"] == 2
    assert report["verification"]["mean_unsupported_when_target_no"] == pytest.approx(
        rows["b2"]["unsupported"]
    )
    assert [r.statement for r in backend.statements] == ["billing", "shipping delay"]


def test_models_without_statement_or_score_batches_fall_back_to_choices():
    class ChoiceOnly:
        remote = True
        name = model_id = "choice-only"
        revision = device = precision = None
        load_time_ms = calibration = None

        def __init__(self):
            self.backend = self
            self.inner = DecisionModel("demo")

        def choose_batch(self, requests):
            return self.inner.choose_batch(requests)

    report = run_benchmark(ChoiceOnly(), mixed_cases())
    rows = {row["id"]: row for row in report["predictions"]}
    assert isinstance(rows["b1"]["value"], bool) and rows["b1"]["method"] == "binary_choice"
    assert set(rows["b1"]["probabilities"]) == {"yes", "no"}
    assert rows["s1"]["level"] == 2 and rows["s1"]["score"] > 1
    assert report["performance"]["measurement"].startswith("end-to-end")


def test_demo_infrastructure_report_and_raw_scores(tmp_path):
    rows = load_dataset(ROOT / "benchmarks/datasets/core.jsonl")
    report = run_benchmark(DecisionModel("demo"), rows, batch_size=4, limit=16, robustness=True)
    assert report["status"] == "infrastructure_only"
    assert report["dataset"]["count"] == 16
    assert report["performance"]["warm_batch_latency_ms"]["samples"] == 3
    choice_rows = sum(1 for row in report["predictions"] if row["kind"] == "choice")
    assert report["robustness"]["choice_order"]["count"] == choice_rows
    assert report["robustness"]["self_consistency"]["count"] == 16
    assert "accuracy" not in report["subjective"]
    assert "state" not in report["predictions"][0]
    assert report["predictions"][0]["raw_scores"] is not None
    # Static result metadata lives once under runtime; rows keep only varying fields.
    assert report["runtime"]["result_metadata"]["latency_scope"] == "batch_wall_including_lock"
    assert "revision" not in report["predictions"][0]["metadata"]
    assert "backend_details" not in report["predictions"][0]["metadata"]
    assert report["predictions"][0]["state_truncated"] is False
    paths = write_report(report, tmp_path / "report.json")
    written = json.loads(Path(paths["json"]).read_text())
    assert written["status"] == "infrastructure_only"
    assert "normalized_probabilities" not in written["predictions"][0]
    assert "normalized_probabilities" in report["predictions"][0]
    assert all(round(p, 6) == p for p in written["predictions"][0]["probabilities"].values())
    assert written["objective"] == report["objective"]
    assert "infrastructure_only" in Path(paths["markdown"]).read_text()


def test_missing_credentials_skip(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ExternalUnavailable, match="not set"):
        create_external_model("gemini")
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_external_model("unknown-provider")


def test_gemini_discloses_self_reported_probabilities(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")

    def handle(request):
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert request.headers["x-goog-api-key"] == "test-placeholder"
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"probabilities": [0.3, 0.7]}'}]}}],
                "modelVersion": "gemini-fixture",
            },
        )

    model = RemoteDecisionModel("gemini", transport=httpx.MockTransport(handle))
    result = model.choose(state="context", question="Answer?", choices=["a", "b"])
    assert result.choice == "b"
    assert "self-reported" in result.metadata["probability_semantics"]
    model.close()


def test_invalid_remote_distribution_fails_without_silent_normalization(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    response = {
        "candidates": [{"content": {"parts": [{"text": '{"probabilities": [0.8, 0.8]}'}]}}],
        "modelVersion": "gemini-fixture",
    }
    model = RemoteDecisionModel(
        "gemini", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    )
    with pytest.raises(ExternalProviderError, match="invalid"):
        model.choose(state="context", question="question", choices=["a", "b"])
    model.close()


def test_remote_thresholds_apply_to_self_reported_distribution(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder")
    response = {
        "candidates": [{"content": {"parts": [{"text": '{"probabilities": [0.8, 0.2]}'}]}}],
        "modelVersion": "gemini-fixture",
    }
    model = RemoteDecisionModel(
        "gemini", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    )
    result = model.choose(
        state="context", question="question", choices=["a", "b"], margin_threshold=0.7
    )
    assert result.abstained
    assert result.confidence == pytest.approx(0.6)
    assert result.model == "gemini-fixture"
    assert result.metadata["remote_state_transmitted"] is True
    model.close()

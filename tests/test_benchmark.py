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


def test_demo_infrastructure_report_and_raw_scores(tmp_path):
    rows = load_dataset(ROOT / "benchmarks/datasets/core.jsonl")
    report = run_benchmark(DecisionModel("demo"), rows, batch_size=4, limit=16, robustness=True)
    assert report["status"] == "infrastructure_only"
    assert report["dataset"]["count"] == 16
    assert report["performance"]["warm_batch_latency_ms"]["samples"] == 3
    assert report["robustness"]["choice_order"]["count"] == 16
    assert "accuracy" not in report["subjective"]
    assert "state" not in report["predictions"][0]
    assert report["predictions"][0]["raw_scores"] is not None
    paths = write_report(report, tmp_path / "report.json")
    assert json.loads(Path(paths["json"]).read_text())["status"] == "infrastructure_only"
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

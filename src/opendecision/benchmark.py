"""Reproducible finite-choice evaluation, with objective and policy results separated."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opendecision.metrics import classification_metrics, percentile, ranking_metrics
from opendecision.schemas import DecisionRequest


class BenchmarkCase(BaseModel):
    """Synthetic or independently collected example with auditable provenance."""

    model_config = ConfigDict(extra="forbid")
    id: str
    family: Literal["objective", "agent_control", "ranking", "subjective", "ambiguous"]
    group_id: str
    split: Literal["train", "calibration", "validation", "test"]
    state: str
    question: str
    choices: list[str]
    target: str | None = None
    expected_abstain: bool = False
    reference_policy: str | None = None
    target_ranking: list[str] | None = None
    variant: str = "base"
    base_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_case(self) -> BenchmarkCase:
        DecisionRequest(state=self.state, question=self.question, choices=self.choices)
        if self.target is not None and self.target not in self.choices:
            raise ValueError("target must be a choice")
        if self.family == "subjective" and self.target is not None and not self.reference_policy:
            raise ValueError("Subjective targets require an explicit reference policy")
        if self.target_ranking is not None and (
            set(self.target_ranking) != set(self.choices)
            or len(self.target_ranking) != len(self.choices)
        ):
            raise ValueError("target_ranking must contain each choice exactly once")
        return self

    def to_request(self) -> DecisionRequest:
        return DecisionRequest(
            state=self.state,
            question=self.question,
            choices=self.choices,
            abstain_threshold=0.55 if self.expected_abstain else None,
            include_raw_scores=True,
        )


def load_dataset(path: str | Path) -> list[BenchmarkCase]:
    """Load JSONL, rejecting duplicate ids and any group crossing split boundaries."""
    cases = []
    identifiers: set[str] = set()
    groups: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = BenchmarkCase.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"Invalid benchmark case at line {number}: {error}") from error
        if case.id in identifiers:
            raise ValueError(f"Duplicate case id: {case.id}")
        if case.group_id in groups and groups[case.group_id] != case.split:
            raise ValueError(f"Split leakage: group {case.group_id} crosses splits")
        identifiers.add(case.id)
        groups[case.group_id] = case.split
        cases.append(case)
    if not cases:
        raise ValueError("Dataset contains no cases")
    return cases


def dataset_hash(cases: list[BenchmarkCase]) -> str:
    """Hash canonical ordered cases, including labels, provenance and split."""
    content = "\n".join(
        json.dumps(case.model_dump(), sort_keys=True, ensure_ascii=False) for case in cases
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (
        "opendecision",
        "torch",
        "transformers",
        "numpy",
        "pydantic",
        "onnxruntime",
        "httpx",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _distribution(result: Any, name: str = "probabilities") -> dict[str, float]:
    return dict(getattr(result, name))


def run_benchmark(
    model: Any,
    cases: list[BenchmarkCase],
    *,
    batch_size: int = 1,
    split: str = "test",
    limit: int | None = None,
    robustness: bool = False,
) -> dict[str, Any]:
    """Run explicitly selected cases; no inference or download happens on import.

    Model inference failures propagate so a partial run cannot masquerade as a
    complete comparison. Predictions retain ids and scores, never input state.
    """
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be positive")
    selected = [case for case in cases if case.split == split]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No examples in split {split!r}")
    results, timings, amortized, rows = [], [], [], []
    process_start = time.process_time()
    start = time.perf_counter()
    for offset in range(0, len(selected), batch_size):
        chunk = selected[offset : offset + batch_size]
        chunk_start = time.perf_counter()
        output = model.choose_batch([case.to_request() for case in chunk])
        elapsed = (time.perf_counter() - chunk_start) * 1000
        if len(output) != len(chunk):
            raise ValueError("Model returned a different result count from request count")
        timings.append(elapsed)
        if offset:
            amortized.extend([elapsed / len(chunk)] * len(chunk))
        results.extend(output)
    total_s = time.perf_counter() - start
    process_s = time.process_time() - process_start
    for case, result in zip(selected, results, strict=True):
        rows.append(
            {
                "id": case.id,
                "base_id": case.base_id,
                "group_id": case.group_id,
                "family": case.family,
                "variant": case.variant,
                "target": case.target,
                "reference_policy": case.reference_policy,
                "target_ranking": case.target_ranking,
                "expected_abstain": case.expected_abstain,
                "choice": result.choice,
                "probabilities": _distribution(result),
                "normalized_probabilities": _distribution(result, "normalized_probabilities"),
                "calibrated_probabilities": getattr(result, "calibrated_probabilities", None),
                "raw_scores": getattr(result, "raw_scores", None),
                "confidence": result.confidence,
                "top_probability": result.top_probability,
                "abstained": result.abstained,
                "reported_latency_ms": result.latency_ms,
                "metadata": getattr(result, "metadata", {}),
            }
        )
    objective = [
        row
        for row in rows
        if row["family"] in ("objective", "agent_control") and row["target"] is not None
    ]
    subjective = [row for row in rows if row["family"] == "subjective"]
    policy = [row for row in subjective if row["reference_policy"] and row["target"] is not None]
    ambiguous = [row for row in rows if row["expected_abstain"]]
    robustness_report: dict[str, Any] = {
        "choice_order": None,
        "self_consistency": None,
        "perturbations": {},
    }
    by_id = {row["id"]: row for row in rows}
    variants: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row["base_id"] and row["base_id"] in by_id:
            variants[row["variant"]].append((by_id[row["base_id"]], row))
    for variant, pairs in sorted(variants.items()):
        paired_objective = [
            (a, b)
            for a, b in pairs
            if a["family"] in ("objective", "agent_control") and a["target"] is not None
        ]
        robustness_report["perturbations"][variant] = {
            "pairs": len(pairs),
            "choice_stability": mean([a["choice"] == b["choice"] for a, b in pairs]),
            "mean_total_variation": mean(
                [
                    sum(
                        abs(a["probabilities"][c] - b["probabilities"][c])
                        for c in a["probabilities"]
                    )
                    / 2
                    for a, b in pairs
                ]
            ),
            "objective_accuracy_delta": mean(
                [
                    float(b["choice"] == b["target"]) - float(a["choice"] == a["target"])
                    for a, b in paired_objective
                ]
            )
            if paired_objective
            else None,
        }
    if robustness:
        permuted = model.choose_batch(
            [
                case.to_request().model_copy(update={"choices": list(reversed(case.choices))})
                for case in selected
            ]
        )
        repeated = model.choose_batch([case.to_request() for case in selected])
        if len(permuted) != len(rows) or len(repeated) != len(rows):
            raise ValueError("Robustness result count mismatch")
        robustness_report["choice_order"] = {
            "count": len(rows),
            "choice_stability": mean(
                [a.choice == b.choice for a, b in zip(results, permuted, strict=True)]
            ),
            "mean_total_variation": mean(
                [
                    sum(abs(a.probabilities[c] - b.probabilities[c]) for c in a.probabilities) / 2
                    for a, b in zip(results, permuted, strict=True)
                ]
            ),
        }
        robustness_report["self_consistency"] = {
            "count": len(rows),
            "choice_stability": mean(
                [a.choice == b.choice for a, b in zip(results, repeated, strict=True)]
            ),
        }
        for row, permuted_result in zip(rows, permuted, strict=True):
            row["reversed_choice_order"] = {
                "choice": permuted_result.choice,
                "probabilities": dict(permuted_result.probabilities),
                "confidence": permuted_result.confidence,
                "abstained": permuted_result.abstained,
            }
    backend = getattr(model, "backend", model)
    calibration = getattr(model, "calibration", None)
    if hasattr(calibration, "model_dump"):
        calibration = calibration.model_dump(mode="json")
    elif calibration is not None:
        calibration = str(calibration)
    cold_ms = getattr(model, "load_time_ms", getattr(backend, "load_time_ms", None))
    infra_only = (
        str(getattr(backend, "name", "")) in ("demo", "lexical", "mock")
        or str(getattr(model, "name", "")) == "demo"
    )
    return {
        "schema_version": 1,
        "status": "infrastructure_only" if infra_only else "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "sha256": dataset_hash(cases),
            "evaluated_sha256": dataset_hash(selected),
            "split": split,
            "count": len(selected),
            "groups": len({c.group_id for c in selected}),
            "base_scenarios": len({c.base_id or c.id for c in selected}),
            "families": dict(Counter(c.family for c in selected)),
            "variants": dict(Counter(c.variant for c in selected)),
            "limit": limit,
            "synthetic": all(c.metadata.get("synthetic") for c in selected),
        },
        "runtime": {
            "model": getattr(backend, "model_id", getattr(model, "name", "unknown")),
            "revision": getattr(backend, "revision", None),
            "backend": getattr(backend, "name", type(backend).__name__),
            "device": str(getattr(backend, "device", "unknown")),
            "precision": getattr(backend, "precision", None),
            "batch_size": batch_size,
            "template": getattr(model, "template", None),
            "calibration": calibration,
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "dependencies": _versions(),
        },
        "objective": classification_metrics(objective),
        "objective_normalized": classification_metrics(
            [{**r, "probabilities": r["normalized_probabilities"]} for r in objective]
        ),
        "by_family": {
            family: classification_metrics([r for r in objective if r["family"] == family])
            for family in ("objective", "agent_control")
        },
        "subjective": {
            "count": len(subjective),
            "policy_labeled_count": len(policy),
            "policy_agreement": mean([r["choice"] == r["target"] for r in policy])
            if policy
            else None,
            "note": "Reference-policy agreement is not moral accuracy. Unlabeled moral cases have no correctness metric.",
        },
        "ranking": ranking_metrics([r for r in rows if r["target_ranking"]]),
        "ambiguity": {
            "count": len(ambiguous),
            "abstention_rate": mean([r["abstained"] for r in ambiguous]) if ambiguous else None,
            "mean_top_probability": mean([r["top_probability"] for r in ambiguous])
            if ambiguous
            else None,
            "threshold": 0.55,
        },
        "performance": {
            "measurement": "end-to-end client-observed remote API"
            if getattr(model, "remote", False)
            else "in-process Python SDK; includes serialization and scoring",
            "cold_model_load_ms": cold_ms,
            "cold_model_load_note": None
            if cold_ms is not None
            else "Not exposed by this model; first request is reported separately.",
            "first_batch_ms": timings[0],
            "warm_batch_latency_ms": {
                "p50": percentile(timings[1:], 0.50),
                "p95": percentile(timings[1:], 0.95),
                "p99": percentile(timings[1:], 0.99),
                "samples": len(timings) - 1,
            },
            "warm_amortized_per_decision_ms": {
                "p50": percentile(amortized, 0.50),
                "p95": percentile(amortized, 0.95),
                "p99": percentile(amortized, 0.99),
            },
            "throughput_decisions_per_s": len(rows) / total_s,
            "total_wall_seconds": total_s,
            "process_cpu_seconds": process_s,
            "process_cpu_percent_of_one_core": 100 * process_s / total_s,
            "process_lifetime_peak_rss_bytes": _rss_bytes(),
            "peak_ram_note": "Process lifetime high-water mark, including previously loaded models; not isolated per model.",
            "model_file_size_bytes": getattr(backend, "model_file_size_bytes", None),
            "mps_utilization": None,
            "local_http_latency_ms": None,
            "model_only_latency_ms": None,
            "unmeasured_note": "Hardware accelerator utilization, isolated model-only and local HTTP timing require separate instrumentation. No estimates substituted.",
        },
        "robustness": robustness_report,
        "predictions": rows,
        "limitations": [
            "Synthetic seed scenarios and deterministic perturbations are correlated, not independently human-validated samples.",
            "Held-out semantic groups reduce template leakage but do not establish real-world generalization.",
            "Abstention threshold is a test configuration, not a validated safety guarantee.",
            "A limited run can omit labels and robustness pairs; AUROC is null where undefined.",
        ]
        + (
            ["Demo backend measures infrastructure only and provides no evidence of model quality."]
            if infra_only
            else []
        ),
    }


def write_report(report: dict[str, Any], output: str | Path) -> dict[str, str]:
    """Write JSON plus a concise Markdown companion; output is a file prefix."""
    prefix = Path(output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = prefix.with_suffix(".json"), prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    runtime, dataset, perf = report["runtime"], report["dataset"], report["performance"]
    lines = [
        "# OpenDecision benchmark report",
        "",
        f"Status: **{report['status']}**. Generated {report['created_at']}.",
        "",
        f"Model `{runtime['model']}`; revision `{runtime['revision']}`; backend `{runtime['backend']}`; device `{runtime['device']}`; batch {runtime['batch_size']}.",
        "",
        f"Dataset: {dataset['count']} decisions, {dataset['base_scenarios']} underlying scenarios, {dataset['groups']} groups, split `{dataset['split']}`. SHA-256 `{dataset['sha256']}`.",
        "",
        "## Objective tasks",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in (
        "count",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "brier_score",
        "ece",
        "coverage",
        "selective_accuracy",
    ):
        lines.append(f"| {metric} | {report['objective'].get(metric, 'not measured')} |")
    lines.extend(
        [
            "",
            "## Separate evaluation families",
            "",
            f"Policy agreement (not moral accuracy): `{report['subjective']['policy_agreement']}` across {report['subjective']['policy_labeled_count']} policy-labeled cases.",
            f"Ranking NDCG: `{report['ranking']['ndcg']}`. Ambiguous-case abstention rate: `{report['ambiguity']['abstention_rate']}`.",
            "",
            "## Timing",
            "",
            f"Measurement: **{perf['measurement']}**.",
            f"Cold model load: `{perf['cold_model_load_ms']}` ms. First batch: `{perf['first_batch_ms']:.3f}` ms.",
            f"Warm batch latency: `{perf['warm_batch_latency_ms']}`. Amortized per decision: `{perf['warm_amortized_per_decision_ms']}`.",
            f"Throughput including first batch: `{perf['throughput_decisions_per_s']:.3f}` decisions/s.",
            f"Process lifetime peak RSS: `{perf['process_lifetime_peak_rss_bytes']}` bytes; this is not isolated model memory.",
            "",
            "## Selective answering",
            "",
            "Thresholds are top-one minus top-two margins.",
            "",
            "| Minimum margin | Coverage | Accuracy when answered |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["objective"].get("coverage_by_margin_threshold", []):
        lines.append(
            f"| {row['min_margin']} | {row['coverage']:.4f} | {row['selective_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "```json",
            json.dumps(report["robustness"], indent=2),
            "```",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "",
            "The companion JSON contains probabilities, raw scores, per-case choice-order probes, reliability bins, confusion matrices, valid-vocabulary AUROC and full runtime metadata.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    return {"json": str(json_path), "markdown": str(md_path)}

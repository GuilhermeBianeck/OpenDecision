"""Summarize a public decision-benchmark run made through /v1/systemone.

The public benchmark's own MIT harness sends each item to the server and
writes one ``results.jsonl`` per tier. This script pools those files with the
item definitions and reports, per tier and overall:

* raw accuracy (an invalid or failed answer counts as wrong);
* chance-corrected accuracy, ``(accuracy - chance) / (1 - chance)``, where
  chance is the mean of ``1 / len(labels)`` over the tier's items, so tiers
  with different option counts can be compared;
* top-label ECE over 10 equal-width bins, and the multi-class Brier score;
* caller-side latency, median and 95th percentile.

Usage::

    python scripts/public_benchmark.py --items HARNESS/datasets/public \\
        --run RUN_DIR --label qwen35 --output benchmarks/reports/public/qwen35.json

``RUN_DIR`` holds one sub-directory per tier (``original``, ``easy``,
``hard``), each with the harness's ``results.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TIERS = ("original", "easy", "hard")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _ece(rows: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(p, ok) for p, ok in rows if low <= p < high or (index == bins - 1 and p == 1.0)]
        if members:
            confidence = sum(p for p, _ in members) / len(members)
            accuracy = sum(ok for _, ok in members) / len(members)
            total += len(members) / len(rows) * abs(confidence - accuracy)
    return total


def _labels(item: dict[str, Any]) -> list[str]:
    if item["question"]["type"] == "noul":
        return ["yes", "no"]
    return [str(label) for label in item["labels"]]


def summarize(items: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["task_id"]: row for row in results}
    correct, chance, calibration, brier, latency = [], [], [], [], []
    for item in items:
        labels = _labels(item)
        chance.append(1 / len(labels))
        row = by_id.get(item["id"])
        ok = bool(row and row.get("correct"))
        correct.append(ok)
        if row and row.get("latency_s") is not None:
            latency.append(row["latency_s"] * 1000)
        probs = (row or {}).get("probs") if row and row.get("valid") else None
        if probs:
            top = max(probs.values())
            calibration.append((top, ok))
            truth = str(item["expected"])
            brier.append(sum((probs.get(k, 0.0) - (k == truth)) ** 2 for k in labels))
    accuracy = sum(correct) / len(items)
    floor = sum(chance) / len(chance)
    return {
        "items": len(items),
        "answered": sum(1 for item in items if item["id"] in by_id),
        "correct": sum(correct),
        "accuracy": accuracy,
        "chance": floor,
        "chance_corrected_accuracy": (accuracy - floor) / (1 - floor),
        "ece": _ece(calibration),
        "brier": sum(brier) / len(brier) if brier else None,
        "latency_ms": {"p50": _percentile(latency, 0.5), "p95": _percentile(latency, 0.95)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--items", required=True, type=Path, help="harness datasets/public")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--harness-commit", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "label": args.label,
        "notes": args.notes,
        "harness_commit": args.harness_commit,
        "tiers": {},
    }
    pooled_items: list[dict[str, Any]] = []
    pooled_results: list[dict[str, Any]] = []
    model = None
    for tier in TIERS:
        items = [json.loads(line) for line in (args.items / f"{tier}.jsonl").open()]
        results = [json.loads(line) for line in (args.run / tier / "results.jsonl").open()]
        model = model or next((r.get("model") for r in results if r.get("model")), None)
        report["tiers"][tier] = summarize(items, results)
        report.setdefault("item_sha256", {})[tier] = hashlib.sha256(
            (args.items / f"{tier}.jsonl").read_bytes()
        ).hexdigest()
        report.setdefault("predictions", []).extend(
            {
                "id": row["task_id"],
                "tier": tier,
                "predicted": row.get("predicted"),
                "correct": bool(row.get("correct")),
                "probabilities": {k: round(v, 4) for k, v in (row.get("probs") or {}).items()},
                "latency_ms": None
                if row.get("latency_s") is None
                else round(row["latency_s"] * 1000, 1),
            }
            for row in results
        )
        pooled_items += items
        pooled_results += results
    report["model"] = model
    report["overall"] = summarize(pooled_items, pooled_results)
    report["predictions"] = report.pop("predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    overall = report["overall"]
    print(
        f"{args.label}: {overall['correct']}/{overall['items']} "
        f"({overall['accuracy']:.1%}), chance-corrected {overall['chance_corrected_accuracy']:.1%}, "
        f"ECE {overall['ece']:.3f}"
    )


if __name__ == "__main__":
    main()

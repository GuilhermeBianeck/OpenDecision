#!/usr/bin/env python3
"""Render benchmark reports side by side as Markdown; no inference, no estimates.

Every number is copied from a report file. Cells are blank when a report does
not contain a metric, so reports from older schema versions still line up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        report["_name"] = path.stem
        reports.append(report)
    return reports


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| --- |" + " ---: |" * (len(headers) - 1)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render(reports: list[dict[str, Any]]) -> str:
    lines = ["# Benchmark report matrix", ""]
    first = reports[0]["dataset"]
    lines.append(
        f"Split `{first['split']}`. Every report below was run on the rows whose dataset hash "
        "it records; a differing hash means a different corpus and the rows are not comparable."
    )
    lines.append("")
    lines.extend(
        table(
            ["Report", "Model", "Revision", "Device", "Rows", "Dataset SHA-256 (prefix)", "Status"],
            [
                [
                    r["_name"],
                    r["runtime"]["model"],
                    (r["runtime"]["revision"] or "")[:12],
                    r["runtime"]["device"],
                    str(r["dataset"]["count"]),
                    r["dataset"]["sha256"][:12],
                    r["status"],
                ]
                for r in reports
            ],
        )
    )
    lines += [
        "",
        "## Objective decisions (pooled objective, agent control, verification, robustness)",
        "",
    ]
    lines.extend(
        table(
            [
                "Report",
                "Count",
                "Accuracy",
                "Balanced",
                "NLL",
                "Brier",
                "ECE",
                "Acc. at margin ≥ 0.5",
                "Coverage at 0.5",
            ],
            [
                [
                    r["_name"],
                    fmt(r["objective"].get("count")),
                    fmt(r["objective"].get("accuracy")),
                    fmt(r["objective"].get("balanced_accuracy")),
                    fmt(r["objective"].get("negative_log_likelihood")),
                    fmt(r["objective"].get("brier_score")),
                    fmt(r["objective"].get("ece")),
                    *next(
                        (
                            [fmt(t["selective_accuracy"]), fmt(t["coverage"])]
                            for t in r["objective"].get("coverage_by_margin_threshold", [])
                            if t["min_margin"] == 0.5
                        ),
                        ["", ""],
                    ),
                ]
                for r in reports
            ],
        )
    )
    families = sorted({f for r in reports for f in r.get("by_family", {})})
    if families:
        lines += ["", "## Accuracy by family", ""]
        lines.extend(
            table(
                ["Report", *families],
                [
                    [r["_name"]]
                    + [
                        fmt(r.get("by_family", {}).get(f, {}).get("accuracy"))
                        + (
                            f" (n={r['by_family'][f]['count']})"
                            if f in r.get("by_family", {})
                            else ""
                        )
                        for f in families
                    ]
                    for r in reports
                ],
            )
        )
    variants = sorted({v for r in reports for v in r.get("by_variant", {})})
    if variants:
        lines += ["", "## Objective accuracy by variant", ""]
        lines.extend(
            table(
                ["Variant", *[r["_name"] for r in reports]],
                [
                    [v] + [fmt(r.get("by_variant", {}).get(v, {}).get("accuracy")) for r in reports]
                    for v in variants
                ],
            )
        )
    lines += ["", "## Other families", ""]
    lines.extend(
        table(
            [
                "Report",
                "Ordinal exact",
                "Ordinal within one",
                "Ordinal |expected error|",
                "Ranking NDCG",
                "Ambiguous abstention",
                "Policy agreement",
                "Statements scored directly",
                "Mean unsupported (target no / yes)",
            ],
            [
                [
                    r["_name"],
                    fmt(r.get("ordinal", {}).get("exact_level_accuracy")),
                    fmt(r.get("ordinal", {}).get("within_one_level_accuracy")),
                    fmt(r.get("ordinal", {}).get("mean_absolute_expected_error")),
                    fmt(r["ranking"].get("ndcg")),
                    fmt(r["ambiguity"].get("abstention_rate")),
                    fmt(r["subjective"].get("policy_agreement")),
                    fmt(r.get("verification", {}).get("statement_scored")),
                    (
                        fmt(r.get("verification", {}).get("mean_unsupported_when_target_no"))
                        + " / "
                        + fmt(r.get("verification", {}).get("mean_unsupported_when_target_yes"))
                    ).strip(" /"),
                ]
                for r in reports
            ],
        )
    )
    lines += ["", "## Timing (in-process, per request, warm)", ""]
    lines.extend(
        table(
            [
                "Report",
                "Cold load ms",
                "p50 ms",
                "p95 ms",
                "p99 ms",
                "Decisions/s",
                "Peak RSS MiB (process lifetime)",
            ],
            [
                [
                    r["_name"],
                    fmt(r["performance"].get("cold_model_load_ms"), 0),
                    fmt(r["performance"]["warm_batch_latency_ms"].get("p50"), 1),
                    fmt(r["performance"]["warm_batch_latency_ms"].get("p95"), 1),
                    fmt(r["performance"]["warm_batch_latency_ms"].get("p99"), 1),
                    fmt(r["performance"].get("throughput_decisions_per_s"), 2),
                    fmt(r["performance"].get("process_lifetime_peak_rss_bytes", 0) / 2**20, 0),
                ]
                for r in reports
            ],
        )
    )
    lines += [
        "",
        "Latency is the wall time of one request batch of size one, including the inference lock; "
        "peak RSS is a process-lifetime high-water mark, not isolated model memory. Reports are "
        "descriptive measurements on correlated synthetic rows, not independent validation.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Benchmark JSON report files")
    parser.add_argument("--output", type=Path, help="Write Markdown here instead of stdout")
    args = parser.parse_args()
    markdown = render(load(args.reports))
    if args.output:
        args.output.write_text(markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

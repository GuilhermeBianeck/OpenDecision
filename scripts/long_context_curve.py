"""Measure accuracy and latency as the state grows to 2k, 8k and 32k tokens.

Each probe is a labelled test-split choice case whose state is hidden inside
filler: numbered archive notes about office logistics that say nothing about
any decision. The case's own state sits at a seeded position (start, middle or
end), so the model must find it as well as answer it. The same cases are used
at every length, so the curve isolates the effect of length.

    python scripts/long_context_curve.py --model lightning \\
        --output benchmarks/reports/long-context/lightning.json

Run one model at a time: the 32k probes need several gigabytes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from opendecision import DecisionModel
from opendecision.benchmark import load_dataset

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (2_048, 8_192, 32_768)
SUBJECTS = (
    "the quarterly offsite",
    "the printer on the third floor",
    "the parking rota",
    "the kitchen rota",
    "the new badge readers",
    "the coffee supplier",
    "the window cleaning",
    "the plant watering",
    "the meeting room bookings",
    "the lift maintenance",
    "the recycling bins",
    "the summer party",
)
VERBS = (
    "was discussed briefly",
    "was postponed to next month",
    "was confirmed without changes",
    "needs a volunteer",
    "was moved to the shared calendar",
    "had no updates",
)


def filler(index: int) -> str:
    rng = random.Random(index)
    return (
        f"Archive note {index}: {rng.choice(SUBJECTS)} {rng.choice(VERBS)}; "
        f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)}."
    )


def embed(state: str, tokens: int, count_tokens, position: str) -> str:
    """Surround ``state`` with filler until the whole text is about ``tokens`` long."""
    notes: list[str] = []
    budget = tokens - count_tokens(state) - 256  # room for question and options
    used, index = 0, 0
    while used < budget:
        note = filler(index)
        notes.append(note)
        used += count_tokens(note) + 1
        index += 1
    cut = {"start": 0, "middle": len(notes) // 2, "end": len(notes)}[position]
    return "\n".join([*notes[:cut], state, *notes[cut:]])


def save(report: dict, output: Path) -> None:
    try:  # MLX backends: peak GPU memory so far
        import mlx.core as mx

        report["peak_memory_bytes"] = mx.get_peak_memory()
    except ImportError:
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=int, default=36)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",")]
    cases = [
        case
        for case in load_dataset(ROOT / "benchmarks" / "datasets" / "core.jsonl")
        if case.split == "test"
        and case.kind == "choice"
        and case.variant == "base"
        and case.family in {"objective", "agent_control"}
        and case.target is not None
    ]
    rng = random.Random(args.seed)
    cases = rng.sample(cases, min(args.cases, len(cases)))
    positions = [("start", "middle", "end")[i % 3] for i in range(len(cases))]

    report = {"model": args.model, "cases": len(cases), "seed": args.seed, "lengths": {}}
    # One load for every length: the state, not max_length, sets the prompt size.
    model = DecisionModel(args.model, max_length=max(lengths))
    count = model.backend.count_tokens
    for length in lengths:
        rows = []
        for case, position in zip(cases, positions):
            state = embed(case.state, length, count, position)
            started = time.perf_counter()
            result = model.choose(state=state, question=case.question, choices=case.choices)
            rows.append(
                {
                    "id": case.id,
                    "position": position,
                    "state_tokens": count(state),
                    "correct": result.choice == case.target,
                    "top_probability": result.top_probability,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        latencies = [row["latency_ms"] for row in rows]
        report["lengths"][str(length)] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "by_position": {
                position: statistics.mean(r["correct"] for r in rows if r["position"] == position)
                for position in ("start", "middle", "end")
                if any(r["position"] == position for r in rows)
            },
            "mean_state_tokens": statistics.mean(row["state_tokens"] for row in rows),
            "latency_ms": {
                "p50": statistics.median(latencies),
                "max": max(latencies),
            },
            "rows": rows,
        }
        summary = report["lengths"][str(length)]
        print(
            f"{args.model} {length}: accuracy {summary['accuracy']:.3f}, "
            f"p50 {summary['latency_ms']['p50']:.0f} ms",
            flush=True,
        )
        # Saved after every length, so a run stopped for memory keeps what finished.
        save(report, args.output)


if __name__ == "__main__":
    main()

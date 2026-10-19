"""Build statement/negation pairs from the corpus's verification records.

Each pair is one state with a statement and its negation. A consistent model
gives ``p(statement) + p(negation) == 1``; how far it misses is the published
probability-consistency check. Only base verification statements the state
supports or contradicts are used: for unaddressed statements both sides can
fairly be unlikely.

Negation is rule-based and deterministic, and every output is listed in the
committed dataset for review. Statements that are already negative are skipped.

    python scripts/generate_negation_pairs.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "datasets" / "core.jsonl"
OUTPUT = ROOT / "benchmarks" / "datasets" / "negation_pairs.jsonl"
AUXILIARIES = ("is", "are", "was", "were", "can", "will", "should", "must")


def negate(statement: str) -> str | None:
    words = statement.rstrip(".").split()
    if "not" in words or "no" in words:
        return None
    for i, word in enumerate(words[1:], 1):
        if word == "has" and words[i + 1 : i + 2] == ["been"]:
            return " ".join([*words[: i + 1], "not", *words[i + 1 :]]) + "."
        if word in AUXILIARIES:
            if words[i + 1 : i + 2] == ["still"]:
                return " ".join([*words[: i + 1], "no", "longer", *words[i + 2 :]]) + "."
            return " ".join([*words[: i + 1], "not", *words[i + 1 :]]) + "."
    for i, word in enumerate(words[1:4], 1):
        if word == "has":
            return " ".join([*words[:i], "does", "not", "have", *words[i + 1 :]]) + "."
        if word.islower() and word.endswith("s") and not word.endswith("ss"):
            stem = word[:-2] if word.endswith(("ches", "shes", "xes", "oes")) else word[:-1]
            return " ".join([*words[:i], "does", "not", stem, *words[i + 1 :]]) + "."
    return None


def main() -> None:
    pairs = []
    for line in SOURCE.open():
        case = json.loads(line)
        kind = case["metadata"].get("verification_type")
        if case["kind"] != "boolean" or case["variant"] != "base" or kind == "unaddressed":
            continue
        negation = negate(case["statement"])
        if negation is None:
            continue
        pairs.append(
            {
                "id": f"{case['id']}.negation",
                "source_id": case["id"],
                "group_id": case["group_id"],
                "split": case["split"],
                "state": case["state"],
                "statement": case["statement"],
                "negation": negation,
                "target": case["target"],
            }
        )
    OUTPUT.write_text("".join(json.dumps(pair, sort_keys=True) + "\n" for pair in pairs))
    print(f"{len(pairs)} pairs -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

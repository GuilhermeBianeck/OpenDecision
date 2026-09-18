"""First run: opendecision pull base. Then this script works offline."""

from opendecision import DecisionModel

model = DecisionModel("base", device="auto")
result = model.choose(
    state="The customer says their card was charged twice.",
    question="Which team should handle this?",
    choices=["billing", "fraud", "technical", "other"],
    abstain_threshold=0.80,
    margin_threshold=0.20,
)
print(result.model_dump_json(indent=2))
if result.abstained:
    print("Escalate for review: this local decision did not meet the configured thresholds.")
# Thresholds on an uncalibrated model are heuristics, not guarantees of correctness.

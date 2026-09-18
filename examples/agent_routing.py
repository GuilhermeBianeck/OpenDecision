"""Independent questions over shared agent state (requires cached base weights)."""

from opendecision import DecisionModel

model = DecisionModel("base")
state = "The user asks for today's exchange rate. No current source has been consulted."
results = model.decide_many(
    state=state,
    questions={
        "Which tool should be used next?": ["web search", "calculator", "none"],
        "Is there enough information to answer now?": ["yes", "no"],
        "Would the next step delete user data?": ["yes", "no"],
    },
    abstain_threshold=0.80,
    margin_threshold=0.15,
)
for question, result in results.items():
    print(question, result.choice, result.confidence, result.abstained)
# These are advisory scores. Application policy still controls side effects.

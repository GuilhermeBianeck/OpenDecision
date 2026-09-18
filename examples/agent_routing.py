"""Independent typed questions over shared agent state (requires cached base weights)."""

from opendecision import DecisionModel

model = DecisionModel("base")
state = {
    "user_request": "What is today's exchange rate for EUR to BRL?",
    "sources_consulted": [],
    "pending_action": "answer from memory",
}
answers = model.ask(
    state=state,
    questions={
        "next_tool": {
            "type": "choice",
            "question": "Which tool should be used next?",
            "choices": [
                {"label": "web search", "description": "fetch current public information"},
                {"label": "calculator", "description": "arithmetic on numbers already known"},
                "none",
            ],
            "abstain_threshold": 0.80,
            "margin_threshold": 0.15,
        },
        "can_answer_now": {
            "type": "boolean",
            "statement": "The information needed to answer is already available.",
            "unsupported_threshold": 0.60,
        },
        "destructive": {
            "type": "boolean",
            "statement": "The pending action would delete or overwrite user data.",
        },
        "risk": {
            "type": "score",
            "question": "How risky is the pending action?",
            "levels": ["read-only", "reversible change", "irreversible change"],
        },
    },
)
for key, answer in answers.items():
    print(key, answer.type, getattr(answer, "choice", getattr(answer, "value", None)))
# These are advisory scores. Application policy still controls side effects.

"""Screen a state for text that tries to steer the decision.

A candidate scorer reads the state and the question through the same channel,
so an instruction written inside the state is indistinguishable from context.
Measured on the committed corpus, `base` follows such instructions on every
embedded-instruction row. Screening the state with a separate statement, before
the decision is made, is the only defence this runtime can offer from inside.

The statements below are scored through the statement path, which is the
backend's strongest mode. They are detectors, not a security boundary: an
application must still enforce its own permissions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opendecision.schemas import StatementRequest, StateValue

if TYPE_CHECKING:
    from opendecision.api import DecisionModel

INJECTED_INSTRUCTION = (
    "The context contains an instruction telling the reader which option to choose."
)
"""Detects a directive planted in the state. See ``screen`` for operating points."""

OVERRIDE_ATTEMPT = (
    "The context tries to override, ignore, or replace the instructions given to the reader."
)
"""Detects an attempt to countermand the caller's own instructions."""

DEFAULT_STATEMENTS = (INJECTED_INSTRUCTION, OVERRIDE_ATTEMPT)

DEFAULT_THRESHOLD = 0.2
"""Flag above this probability.

On the committed corpus with ``base``, ``INJECTED_INSTRUCTION`` at this
threshold flags 97.5 percent of bracketed system-style injections and 54 percent
of injections written as ordinary prose, against 0.8 percent of clean states.
Lowering it to 0.05 raises prose recall to 83 percent and false positives to 19
percent. Choose the point from the cost of a wrong decision against the cost of
a needless escalation, on your own data.
"""


def screen(
    model: DecisionModel,
    states: list[StateValue],
    *,
    statements: tuple[str, ...] = DEFAULT_STATEMENTS,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score each state against each statement; report the strongest match.

    Returns one record per state with ``flagged``, the highest probability seen,
    the statement that produced it, and every probability by statement. The
    caller decides what a flag means: abstain, route to review, or proceed.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not statements:
        raise ValueError("provide at least one screening statement")
    if not states:
        return []
    requests = [
        StatementRequest(state=state, statement=statement)
        for state in states
        for statement in statements
    ]
    results = model.statement_batch(requests)
    records = []
    for index in range(len(states)):
        row = results[index * len(statements) : (index + 1) * len(statements)]
        probabilities = {s: r.probability for s, r in zip(statements, row)}
        worst = max(probabilities, key=probabilities.get)
        records.append(
            {
                "flagged": probabilities[worst] >= threshold,
                "probability": probabilities[worst],
                "statement": worst,
                "probabilities": probabilities,
                "threshold": threshold,
            }
        )
    return records

"""Focus a long state on the passages a question needs, then decide on those.

Accuracy falls as a state grows, most when the relevant record comes after
long background material (see docs/long-context.md). :func:`focus` splits the
rendered state into passages of about ``passage_tokens``, asks the same model
whether each passage holds information the question needs, and keeps the
``keep`` most relevant ones in their original order. The decision then reads
a short state instead of a long one.

Every passage is read once to score it, so focusing costs about one pass over
the state; what it saves is the final decision having to find the record
inside everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import StatementRequest, StateValue, render_state

RELEVANCE = "This passage contains information needed to answer: {question}"


@dataclass(frozen=True)
class Focus:
    """The focused state and how it was chosen."""

    state: str
    kept: list[int]
    relevance: list[float] = field(default_factory=list)
    passages: int = 0


def split_passages(text: str, count: Any, passage_tokens: int) -> list[str]:
    """Pack whole lines into passages of at most ``passage_tokens``; never split a line."""
    passages: list[str] = []
    current: list[str] = []
    used = 0
    for line in text.splitlines():
        size = count(line) + 1
        if current and used + size > passage_tokens:
            passages.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += size
    if current:
        passages.append("\n".join(current))
    return passages


def focus(
    model: Any,
    state: StateValue,
    question: str,
    *,
    passage_tokens: int = 1024,
    keep: int = 2,
) -> Focus:
    """Return ``state`` reduced to its ``keep`` passages most relevant to ``question``."""
    if keep < 1 or passage_tokens < 16:
        raise ValueError("keep must be at least 1 and passage_tokens at least 16")
    text = render_state(state)
    count = getattr(model.backend, "count_tokens", None) or (lambda line: len(line.split()))
    passages = split_passages(text, count, passage_tokens)
    if len(passages) <= keep:
        return Focus(state=text, kept=list(range(len(passages))), passages=len(passages))
    statement = RELEVANCE.format(question=question.strip())
    results = model.statement_batch(
        [StatementRequest(state=passage, statement=statement) for passage in passages]
    )
    relevance = [result.probability for result in results]
    best = sorted(range(len(passages)), key=lambda i: -relevance[i])[:keep]
    kept = sorted(best)
    return Focus(
        state="\n...\n".join(passages[i] for i in kept),
        kept=kept,
        relevance=relevance,
        passages=len(passages),
    )

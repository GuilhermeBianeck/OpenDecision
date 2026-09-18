"""Which backend suits which question, and the measurement behind each answer.

Backend choice is the largest single lever in this library and the easiest to
get wrong, because the aliases describe model families rather than tasks. The
same question asked of two backends has swung accuracy from 0.15 to 0.85 and
from 0 of 8 to 6 of 8 in the committed benchmarks. The table below records what
was measured, so the choice can be made from evidence instead of intuition.

Every figure comes from the test split of the committed corpus
(``benchmarks/reports/*-mps-test.json``) unless the note says otherwise. These
are correlated synthetic rows: they separate the backends clearly, and they do
not predict accuracy on your own distribution. Measure before relying on one.
"""

from __future__ import annotations

from dataclasses import dataclass

from opendecision.errors import BackendError


@dataclass(frozen=True)
class TaskProfile:
    """A task shape, the backend measured best for it, and why."""

    task: str
    backend: str
    summary: str
    evidence: str
    runners_up: tuple[str, ...] = ()

    def describe(self) -> str:
        alternatives = f" Runners-up: {', '.join(self.runners_up)}." if self.runners_up else ""
        return f"{self.task}: use '{self.backend}'. {self.summary} {self.evidence}{alternatives}"


TASKS: dict[str, TaskProfile] = {
    "verification": TaskProfile(
        task="verification",
        backend="base",
        summary="Deciding whether a statement is supported by the state.",
        evidence="Verification family: base 0.963, tiny 0.932, smart 0.812, decoder 0.696, "
        "multilingual 0.446.",
        runners_up=("tiny",),
    ),
    "relevance": TaskProfile(
        task="relevance",
        backend="multilingual",
        summary="Picking which candidate best matches the state, by topic rather than by "
        "anything stated outright.",
        evidence="Matching 20 headlines to the right one of 15 brands: multilingual 0.85 "
        "top-one against 0.25 for base scoring the same choice, and 0.15 for base scoring "
        "each brand as its own rubric.",
        runners_up=("decoder",),
    ),
    "ranking": TaskProfile(
        task="ranking",
        backend="multilingual",
        summary="Ordering every candidate, not just choosing one.",
        evidence="Ranking family NDCG: multilingual 0.983, decoder 0.982, tiny 0.940, "
        "smart 0.926, base 0.867.",
        runners_up=("decoder",),
    ),
    "rubric": TaskProfile(
        task="rubric",
        backend="smart",
        summary="Rating the state on ordered levels.",
        evidence="Ordinal family exact level: smart 0.611, decoder 0.509, base 0.421, "
        "multilingual 0.370, tiny 0.361.",
        runners_up=("decoder",),
    ),
    "policy": TaskProfile(
        task="policy",
        backend="base",
        summary="Applying a rule stated in the state to the facts also stated there.",
        evidence="Agent-control family: base 0.750, multilingual 0.679, tiny 0.635, "
        "smart 0.628, decoder 0.487. Decompose the rule into its condition first; that "
        "moves negated facts from 0.533 to 0.933 (docs/patterns.md).",
        runners_up=("multilingual",),
    ),
    "stated_facts": TaskProfile(
        task="stated_facts",
        backend="decoder",
        summary="Choosing between candidates whose merit is written out in the candidate "
        "text, such as annotations computed by your own code.",
        evidence="Mate-in-one over annotated moves: decoder 6 of 8 from bare notation, "
        "multilingual 0 of 8 bare and 8 of 8 only when the answer appeared verbatim in the "
        "text. In full games the decoder scored +7.7 Elo against multilingual's -98.9 on "
        "the same annotations.",
    ),
    "wide_choice": TaskProfile(
        task="wide_choice",
        backend="decoder",
        summary="Many candidates for one question, where a cross-encoder would pay a "
        "forward pass each.",
        evidence="Twelve-option routing on unperturbed states: decoder 0.917, multilingual "
        "0.750, base 0.583, tiny 0.333. The decoder caps at 26 options; use "
        "DecisionModel.choose_wide beyond that.",
        runners_up=("multilingual",),
    ),
    "many_questions": TaskProfile(
        task="many_questions",
        backend="decoder",
        summary="Several questions about one state, where the state should be read once.",
        evidence="Thirteen boolean questions over a 1,500-token state: decoder 2.8 s, "
        "base 41 s. The decoder keeps one KV-cache prefix per distinct state.",
    ),
    "classification": TaskProfile(
        task="classification",
        backend="decoder",
        summary="General labelling of a state into categories.",
        evidence="Objective family: decoder 0.618, multilingual 0.582, base 0.569, "
        "smart 0.538, tiny 0.346. Pooled across all objective families base leads at "
        "0.675, so measure both on your own labels.",
        runners_up=("multilingual", "base"),
    ),
}


def recommend(task: str) -> TaskProfile:
    """Return the measured recommendation for a task shape."""
    profile = TASKS.get(task)
    if profile is None:
        raise BackendError(
            f"Unknown task {task!r}. Choose one of: {', '.join(sorted(TASKS))}. "
            "Run 'opendecision tasks' for the measurements behind each."
        )
    return profile


def profiles() -> list[TaskProfile]:
    """Every task profile, in a stable order."""
    return [TASKS[key] for key in sorted(TASKS)]

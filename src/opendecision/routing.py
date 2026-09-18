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
    """A task shape, the backend measured best for it, and why.

    ``portable`` is the best backend that runs anywhere. It differs from
    ``backend`` when the strongest option needs MLX, which is Apple silicon only.
    """

    task: str
    backend: str
    summary: str
    evidence: str
    runners_up: tuple[str, ...] = ()
    portable: str | None = None

    @property
    def portable_backend(self) -> str:
        """The recommendation for a machine without MLX."""
        return self.portable or self.backend

    def describe(self) -> str:
        alternatives = f" Runners-up: {', '.join(self.runners_up)}." if self.runners_up else ""
        elsewhere = f" Off Apple silicon use '{self.portable}'." if self.portable else ""
        return (
            f"{self.task}: use '{self.backend}'. {self.summary} "
            f"{self.evidence}{alternatives}{elsewhere}"
        )


TASKS: dict[str, TaskProfile] = {
    "verification": TaskProfile(
        task="verification",
        backend="qwen35",
        summary="Deciding whether a statement is supported by the state.",
        evidence="Verification family: qwen35 0.989, qwen35_4b 0.983, base 0.963, tiny 0.932, "
        "smart 0.812, decoder 0.696, lfm25 0.594, multilingual 0.446.",
        runners_up=("qwen35_4b", "base"),
        portable="base",
    ),
    "relevance": TaskProfile(
        task="relevance",
        backend="multilingual",
        summary="Picking which candidate best matches the state, by topic rather than by "
        "anything stated outright.",
        evidence="Matching 20 headlines to the right one of 15 brands: multilingual 0.85 "
        "top-one, qwen35 0.80, base 0.25 scoring the same choice, base 0.15 scoring each "
        "brand as its own rubric. Reranking is what this checkpoint was trained for.",
        runners_up=("qwen35",),
    ),
    "ranking": TaskProfile(
        task="ranking",
        backend="multilingual",
        summary="Ordering every candidate, not just choosing one.",
        evidence="Ranking family NDCG: qwen35 0.984, multilingual 0.983, decoder 0.982, "
        "lfm25 0.952, qwen35_4b 0.942, tiny 0.940, smart 0.926, base 0.867. The top three "
        "are inside the noise of 52 rows; multilingual is the smallest of them.",
        runners_up=("qwen35", "decoder"),
    ),
    "rubric": TaskProfile(
        task="rubric",
        backend="qwen35_4b",
        summary="Rating the state on ordered levels.",
        evidence="Ordinal family exact level: qwen35_4b 0.769, smart 0.611, decoder 0.509, "
        "lfm25 0.509, qwen35 0.481, base 0.421, multilingual 0.370, tiny 0.361.",
        runners_up=("smart",),
        portable="smart",
    ),
    "policy": TaskProfile(
        task="policy",
        backend="qwen35",
        summary="Applying a rule stated in the state to the facts also stated there.",
        evidence="Agent-control family: qwen35 0.814, qwen35_4b 0.801, base 0.750, "
        "multilingual 0.679, tiny 0.635, smart 0.628, decoder 0.487; lfm25 scores 0.840 "
        "here but 0.596 overall with the worst calibration of any backend. On negated "
        "facts qwen35_4b answers every unperturbed scenario correctly against 0.500 for "
        "base, so decomposition (docs/patterns.md) matters less than it did.",
        runners_up=("qwen35_4b", "base"),
        portable="base",
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
        evidence="Twelve-option routing on unperturbed states: decoder 0.917, qwen35 0.917, "
        "qwen35_4b 0.833, multilingual 0.750, base 0.583, lfm25 0.417, tiny 0.333. Both "
        "decoders cap at 26 options; use DecisionModel.choose_wide beyond that.",
        runners_up=("qwen35",),
    ),
    "many_questions": TaskProfile(
        task="many_questions",
        backend="decoder",
        summary="Several questions about one state, where the state should be read once.",
        evidence="Thirteen boolean questions over a 1,500-token state: decoder 0.88 s, "
        "qwen35 1.02 s, base 4.18 s. Both decoders reuse an encoded prefix; the "
        "cross-encoders re-read the state for every candidate.",
        runners_up=("qwen35",),
    ),
    "classification": TaskProfile(
        task="classification",
        backend="qwen35",
        summary="General labelling of a state into categories.",
        evidence="Objective family: qwen35_4b 0.854, qwen35 0.813, decoder 0.618, "
        "multilingual 0.582, base 0.569, smart 0.538, lfm25 0.462, tiny 0.346. Pooled over "
        "every objective family the order holds: qwen35_4b 0.853, qwen35 0.799, base 0.675.",
        runners_up=("qwen35_4b", "base"),
        portable="base",
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

"""Decision-trained checkpoints, scored with the exact prompt they were trained on.

A general instruction model answers whatever wording it is given; a model
fine-tuned for decisions has learned one prompt, and any other costs accuracy.
This backend reproduces that prompt for the H2O-Lightning family:

* a fixed system turn, then one user turn of ``record:``, ``question:`` and
  ``options:`` lines, each option written ``A) name: description``;
* the chat template with thinking closed, and ``Answer:`` already written;
* the answer is the next-token distribution over `` A``, `` B``, ... .

A yes/no statement is a two-option question, ``true`` and ``false``. Prefix
caching, state truncation, the memory guard and the bounded MLX cache are the
general MLX backend's.
"""

from __future__ import annotations

import math
from typing import Any

from opendecision.backends.decoder import LETTERS
from opendecision.backends.mlx_decoder import MLXDecoderBackend
from opendecision.errors import BackendError

SYSTEM = (
    "You are a decision engine. You read a record and answer one question about it by "
    "choosing exactly one option. Reply with a single letter."
)
PREFILL = "Answer:"
STATEMENT_OPTIONS = ("true: the statement holds", "false: it does not")
# The statement contract has no "unknown" answer; this keeps the triple finite
# while giving it no probability mass.
NO_UNKNOWN = -1e4
# Unlike a general 2B model, this checkpoint uses long states: on the public
# benchmark an 8,192-token context answers 204 of 231 items against 197 at
# 2,048, at a measured 4.46 GB peak.
DEFAULT_MAX_LENGTH = 8192


class DecisionMLXBackend(MLXDecoderBackend):
    label_prefix = " "

    def __init__(self, *, max_length: int | None = None, **kwargs: Any):
        super().__init__(
            max_length=DEFAULT_MAX_LENGTH if max_length is None else max_length, **kwargs
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "prompt_format": "record-question-options-v1",
            "scoring": "next-token log-probability of the option label after 'Answer:'",
            "statement_options": list(STATEMENT_OPTIONS),
        }

    @staticmethod
    def _question_text(question: str, options: list[str]) -> str:
        lines = [f"{LETTERS[i]}) {option}" for i, option in enumerate(options)]
        return f"question: {question}\noptions:\n" + "\n".join(lines)

    def _candidates(self, request: Any) -> list[str]:
        # Trained on "name: description" lines; a bare option repeats its name.
        texts = []
        for choice, text in zip(request.choices, request.candidate_texts):
            label = choice if isinstance(choice, str) else choice.label
            texts.append(text if text != label else f"{label}: {label}")
        return texts

    @staticmethod
    def _prompt_text(state: str, question: str, images: int = 0) -> str:
        # Images go first in the user turn, one vision slot each.
        slots = "<|vision_start|><|image_pad|><|vision_end|>" * images
        return (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{slots}record: {state}\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{PREFILL}"
        )

    def _prompt_ids(self, state: str, question: str) -> list[int]:
        return self._encode(self._prompt_text(state, question))

    def score_statements(self, requests: list[Any]) -> list[list[float]]:
        """Return [true, unknown, false] log-probabilities; unknown is never chosen."""
        if not requests:
            return []
        with self._lock:
            self.load()
            self._truncated_states = 0
            results: list[list[float] | None] = [None] * len(requests)
            for state_text, indexes in self._grouped([r.state_text for r in requests]).items():
                suffixes = [
                    self._question_text(requests[index].statement, list(STATEMENT_OPTIONS))
                    for index in indexes
                ]
                rows = self._guard(lambda: self._run_group(state_text, suffixes))
                for index, row in zip(indexes, rows):
                    results[index] = [row[0], NO_UNKNOWN, row[1]]
            self._warn_truncation("statements")
            rows = [row for row in results if row is not None]
            if len(rows) != len(requests) or not all(
                math.isfinite(value) for row in rows for value in row
            ):
                raise BackendError("Backend returned missing or non-finite statement scores.")
            return rows

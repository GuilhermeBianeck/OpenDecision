"""The decision-trained backend's prompt contract, without loading weights."""

from __future__ import annotations

import pytest

from opendecision import DecisionModel
from opendecision.backends.decision_mlx import NO_UNKNOWN, DecisionMLXBackend
from opendecision.registry import create_backend
from opendecision.schemas import ChoiceOption, DecisionRequest, StatementRequest


class Characters:
    """Tokenizer stand-in: one id per character, so prompts round-trip exactly."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


@pytest.fixture
def backend():
    value = create_backend("lightning")
    assert isinstance(value, DecisionMLXBackend)
    value._tokenizer = Characters()
    return value


def test_the_alias_and_catalog_entry_resolve():
    assert create_backend("h2o").model_id == "netprojects/h2o-lightning-4b-mlx-q4"


def test_options_follow_the_training_layout(backend):
    request = DecisionRequest(
        state="s",
        question="Which team?",
        choices=["sales", ChoiceOption(label="billing", description="charges and refunds")],
    )
    options = backend._candidates(request)
    assert options == ["sales: sales", "billing: charges and refunds"]
    assert backend._question_text("Which team?", options) == (
        "question: Which team?\noptions:\nA) sales: sales\nB) billing: charges and refunds"
    )


def test_the_prompt_closes_thinking_and_prefills_the_answer(backend):
    ids = backend._prompt_ids("the record", "question: q\noptions:\nA) a: a")
    text = "".join(map(chr, ids))
    assert text.startswith("<|im_start|>system\nYou are a decision engine.")
    assert "<|im_start|>user\nrecord: the record\nquestion: q\noptions:\nA) a: a<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def test_long_states_are_kept_by_default():
    assert create_backend("lightning").max_length == 8192
    assert create_backend("lightning", max_length=2048).max_length == 2048


def test_labels_are_read_with_a_leading_space():
    assert DecisionMLXBackend.label_prefix == " "


def test_statements_are_true_or_false_questions(backend, monkeypatch):
    seen = []
    monkeypatch.setattr(backend, "load", lambda: None)

    def run_group(state_text, suffixes):
        seen.extend(suffixes)
        return [[-0.1, -2.5] for _ in suffixes]

    monkeypatch.setattr(backend, "_run_group", run_group)
    rows = backend.score_statements([StatementRequest(state="s", statement="It is paid.")])
    assert seen == [
        "question: It is paid.\noptions:\nA) true: the statement holds\nB) false: it does not"
    ]
    assert rows == [[-0.1, NO_UNKNOWN, -2.5]]
    result = DecisionModel(backend=backend).statement_batch(
        [StatementRequest(state="s", statement="It is paid.")]
    )[0]
    assert result.probability > 0.9
    assert result.unsupported == pytest.approx(0, abs=1e-6)

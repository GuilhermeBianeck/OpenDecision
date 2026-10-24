"""Focusing a long state on the passages a question needs."""

from __future__ import annotations

import pytest
from test_core import StatementBackend

from opendecision import DecisionModel
from opendecision.focus import RELEVANCE, focus, split_passages


def words(line):
    return len(line.split())


def test_passages_pack_whole_lines_up_to_the_budget():
    text = "\n".join(["one two three"] * 5)  # 3 words + 1 per line
    passages = split_passages(text, words, passage_tokens=8)
    assert passages == ["one two three\none two three"] * 2 + ["one two three"]
    long_line = "x " * 50
    assert split_passages(long_line, words, 8) == [long_line]  # a line is never split


class Relevant(StatementBackend):
    """Marks a passage relevant when it mentions the invoice."""

    def score_statements(self, requests):
        self.statements.extend(requests)
        return [
            [4.0, -9.0, -4.0] if "invoice" in r.state_text else [-4.0, -9.0, 4.0] for r in requests
        ]


def test_focus_keeps_the_relevant_passages_in_order():
    lines = [f"Archive note {i}: nothing happened." for i in range(40)]
    lines[25] = "The invoice 4471 was paid on 2 October."
    lines[5] = "The invoice total is 1,240 EUR."
    backend = Relevant()
    result = focus(
        DecisionModel(backend=backend),
        "\n".join(lines),
        "Was the invoice paid?",
        passage_tokens=30,
        keep=2,
    )
    assert result.passages > 2
    assert "1,240 EUR" in result.state and "2 October" in result.state
    assert result.state.index("1,240") < result.state.index("2 October")  # original order
    assert backend.statements[0].statement == RELEVANCE.format(question="Was the invoice paid?")
    assert len(result.relevance) == result.passages


def test_short_states_are_returned_whole_without_scoring():
    backend = Relevant()
    result = focus(DecisionModel(backend=backend), "a short state", "q?", keep=2)
    assert result.state == "a short state" and result.kept == [0]
    assert backend.statements == []


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        focus(DecisionModel("demo"), "s", "q", keep=0)

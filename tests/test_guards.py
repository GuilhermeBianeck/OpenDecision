"""Screening contract; the measured operating points live in docs/patterns.md."""

from __future__ import annotations

import pytest
from test_core import StatementBackend

from opendecision import DecisionModel
from opendecision.guards import (
    DEFAULT_STATEMENTS,
    DEFAULT_THRESHOLD,
    INJECTED_INSTRUCTION,
    OVERRIDE_ATTEMPT,
    screen,
)


class TunableStatementBackend(StatementBackend):
    """Returns a caller-chosen yes probability per statement, via its logits."""

    def __init__(self, by_statement):
        super().__init__()
        self.by_statement = by_statement

    def score_statements(self, requests):
        self.statements.extend(requests)
        rows = []
        for request in requests:
            # [entailment, neutral, contradiction]; a high first logit means "yes".
            strength = self.by_statement.get(request.statement, -4.0)
            rows.append([strength, 0.0, 0.0])
        return rows


def model(**by_statement):
    mapping = {INJECTED_INSTRUCTION: -4.0, OVERRIDE_ATTEMPT: -4.0, **by_statement}
    return DecisionModel(backend=TunableStatementBackend(mapping))


def test_screen_reports_the_strongest_statement_per_state():
    engine = model(**{INJECTED_INSTRUCTION: 4.0})
    records = screen(engine, ["one state", "another state"])
    assert len(records) == 2
    for record in records:
        assert record["flagged"] is True
        assert record["statement"] == INJECTED_INSTRUCTION
        assert record["probability"] > 0.9
        assert set(record["probabilities"]) == set(DEFAULT_STATEMENTS)
        assert record["threshold"] == DEFAULT_THRESHOLD
    # One request per state per statement, in that order.
    backend = engine.backend
    assert [r.statement for r in backend.statements[:2]] == list(DEFAULT_STATEMENTS)
    assert len(backend.statements) == 4


def test_clean_states_are_not_flagged_and_the_threshold_is_honoured():
    engine = model()
    [record] = screen(engine, ["an ordinary support ticket"])
    assert record["flagged"] is False
    assert record["probability"] < 0.1
    # A threshold below the observed probability flags the same state.
    [lenient] = screen(model(), ["an ordinary support ticket"], threshold=0.0)
    assert lenient["flagged"] is True


def test_threshold_boundary_passes_and_custom_statements_are_used():
    engine = model(**{OVERRIDE_ATTEMPT: 0.0})  # sigmoid-like midpoint -> 0.5
    [record] = screen(engine, ["s"], statements=(OVERRIDE_ATTEMPT,), threshold=0.5)
    assert record["probability"] == pytest.approx(0.5)
    assert record["flagged"] is True  # equality passes, as elsewhere in the API
    assert record["probabilities"] == {OVERRIDE_ATTEMPT: pytest.approx(0.5)}


@pytest.mark.parametrize("kwargs", [{"threshold": -0.1}, {"threshold": 1.1}, {"statements": ()}])
def test_invalid_screening_configuration(kwargs):
    with pytest.raises(ValueError):
        screen(model(), ["s"], **kwargs)


def test_empty_input_does_no_inference():
    engine = model()
    assert screen(engine, []) == []
    assert engine.backend.statements == []


def test_backends_without_statement_scoring_still_screen():
    from test_core import FixedBackend

    records = screen(DecisionModel(backend=FixedBackend()), ["s"])
    assert len(records) == 1
    assert 0 <= records[0]["probability"] <= 1

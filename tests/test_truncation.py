"""The shared route refuses over-long requests; the native API still trims."""

from __future__ import annotations

import threading

import pytest
from test_core import StatementBackend

from opendecision import DecisionModel
from opendecision.backends.truncation import check_truncation, refuse_truncation
from opendecision.errors import ContextOverflow


class Trimming(StatementBackend):
    """Pretends every state is too long, so it would trim (or refuse) each time."""

    def score_batch(self, requests):
        check_truncation(5000, 2048)
        return [[1.0] + [0.0] * (len(r.choices) - 1) for r in requests]


REQUEST = {
    "state": "a very long contract",
    "model": "m",
    "questions": {
        "q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
    },
}


def test_trimming_is_allowed_unless_refused():
    check_truncation(5000, 2048)  # no-op by default
    with refuse_truncation(), pytest.raises(ContextOverflow) as raised:
        check_truncation(5000, 2048)
    assert (raised.value.tokens, raised.value.limit) == (5000, 2048)
    check_truncation(5000, 2048)  # restored after the block


def test_refusal_is_per_thread():
    seen = []
    with refuse_truncation():
        worker = threading.Thread(target=lambda: seen.append(check_truncation(5000, 2048)))
        worker.start()
        worker.join()
    assert seen == [None]


def test_the_shared_route_answers_400_max_tokens_exceeded():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from opendecision.server import create_app

    engine = DecisionModel(backend=Trimming())
    with TestClient(create_app("demo", decision_model=engine)) as client:
        response = client.post("/v1/systemone", json=REQUEST)
        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "max_tokens_exceeded",
            "message": response.json()["detail"]["message"],
            "tokens": 5000,
            "limit": 2048,
        }
        native = client.post(
            "/v1/decide", json={"state": "s", "question": "Which?", "choices": ["a", "b"]}
        )
        assert native.status_code == 200


def test_the_decoder_prompt_fitter_refuses_instead_of_trimming():
    from opendecision.registry import create_backend

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

    backend = create_backend("lightning", max_length=512)
    backend._tokenizer = Characters()
    state = "x" * 2000
    rows = backend._fit_prompts(state, ["question: q\noptions:\nA) a: a"])
    assert max(map(len, rows)) <= 512  # trimmed by default
    with refuse_truncation(), pytest.raises(ContextOverflow) as raised:
        backend._fit_prompts(state, ["question: q\noptions:\nA) a: a"])
    assert raised.value.limit == 512 and raised.value.tokens > 2000

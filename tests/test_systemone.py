"""The shared /v1/systemone wire format, end to end through the HTTP server."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_core import StatementBackend

from opendecision import DecisionModel
from opendecision.server import create_app

# Copied in shape from the public specification: one question of each type.
SPEC_REQUEST = {
    "state": "Hi, I have been trying to connect my Stripe account for 3 days and it keeps failing.",
    "model": "any-model-name-an-sdk-sends",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, refunds and invoices",
                "integrations": "Connecting third-party accounts",
                "sales": None,
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer appear?",
            "criteria": ["calm", "irritated", "angry"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity.",
        },
    },
}


@pytest.fixture
def client():
    with TestClient(create_app("demo")) as value:
        yield value


def test_a_specification_request_succeeds_unmodified(client):
    response = client.post("/v1/systemone", json=SPEC_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"model", "answers"}
    assert list(body["answers"]) == ["department", "frustration", "is_urgent"]

    choice = body["answers"]["department"]
    assert choice["type"] == "choice"
    assert choice["choice"] in SPEC_REQUEST["questions"]["department"]["criteria"]
    assert set(choice["probabilities"]) == {"billing", "integrations", "sales"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1)
    assert 0 <= choice["confidence"] <= 1

    score = body["answers"]["frustration"]
    assert score["type"] == "score"
    assert score["legend"] == {"0": "calm", "1": "irritated", "2": "angry"}
    assert list(score["probabilities"]) == ["0", "1", "2"]
    assert 0 <= score["score"] <= 2
    assert score["score"] == pytest.approx(
        sum(int(k) * p for k, p in score["probabilities"].items())
    )

    noul = body["answers"]["is_urgent"]
    assert noul == {"type": "noul", "noul": noul["noul"]}
    assert 0 <= noul["noul"] <= 1


def test_the_answering_checkpoint_is_reported_not_the_requested_name(client):
    body = client.post("/v1/systemone", json=SPEC_REQUEST).json()
    assert body["model"] == "opendecision/token-overlap-demo"


def test_option_descriptions_reach_the_model_and_answers_key_by_option_name():
    seen = []

    class Recording(StatementBackend):
        def score_batch(self, requests):
            seen.extend(requests)
            return [[1.0] + [0.0] * (len(r.choices) - 1) for r in requests]

    engine = DecisionModel(backend=Recording())
    with TestClient(create_app("demo", decision_model=engine)) as client:
        body = client.post(
            "/v1/systemone",
            json={
                "state": "s",
                "model": "m",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick",
                        "criteria": {"alpha": "the first", "beta": None},
                    }
                },
            },
        ).json()
    assert body["answers"]["q"]["choice"] == "alpha"
    assert seen[0].labels == ["alpha", "beta"]
    assert seen[0].candidate_texts == ["alpha: the first", "beta"]


def test_noul_uses_statement_scoring_when_the_backend_has_it():
    engine = DecisionModel(backend=StatementBackend([3.0, 0.0, -3.0]))
    with TestClient(create_app("demo", decision_model=engine)) as client:
        body = client.post(
            "/v1/systemone",
            json={
                "state": "s",
                "model": "m",
                "questions": {"q": {"type": "noul", "instructions": "It is true."}},
            },
        ).json()
    assert body["answers"]["q"]["noul"] > 0.99
    assert engine.backend.statements[0].statement == "It is true."


def test_structured_state_is_accepted(client):
    request = {**SPEC_REQUEST, "state": {"message": "billing issue", "order_id": "A-104"}}
    assert client.post("/v1/systemone", json=request).status_code == 200
    request = {**SPEC_REQUEST, "state": ["first message", "second message"]}
    assert client.post("/v1/systemone", json=request).status_code == 200


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.pop("questions"),
        lambda r: r.pop("model"),
        lambda r: r.update(questions={}),
        lambda r: r["questions"]["department"].update(type="pick"),
        lambda r: r["questions"]["department"].update(criteria={"only": None}),
        lambda r: r["questions"]["frustration"].update(criteria=["one"]),
        lambda r: r["questions"]["is_urgent"].update(instructions="  "),
    ],
)
def test_malformed_requests_are_rejected_with_422(client, mutation):
    import copy

    request = copy.deepcopy(SPEC_REQUEST)
    mutation(request)
    assert client.post("/v1/systemone", json=request).status_code == 422


def test_the_route_is_documented(client):
    assert "/v1/systemone" in client.get("/openapi.json").json()["paths"]

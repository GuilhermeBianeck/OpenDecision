"""Contract and resource-bound checks for the local HTTP service."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from opendecision import DecisionModel
from opendecision.server import create_app
from opendecision.server.app import MAX_BODY_BYTES

REQUEST = {
    "state": "Billing needs billing support.",
    "question": "Which team?",
    "choices": ["billing", "technical"],
}


@pytest.fixture
def client():
    with TestClient(create_app("demo")) as value:
        yield value


def test_resident_engine_created_once(monkeypatch):
    calls = []
    engine = DecisionModel("demo")

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return engine

    monkeypatch.setattr("opendecision.server.app.DecisionModel", factory)
    with TestClient(create_app("demo")) as client:
        for _ in range(2):
            assert client.post("/v1/decide", json=REQUEST).status_code == 200
        assert client.get("/health").json()["telemetry"] is False
    assert len(calls) == 1


def test_choose_and_rank_contract(client):
    result = client.post("/v1/decide", json={**REQUEST, "include_raw_scores": True})
    assert result.status_code == 200
    body = result.json()
    assert body["choice"] == "billing"
    assert sum(body["probabilities"].values()) == pytest.approx(1)
    assert body["calibrated_probabilities"] is None
    assert body["raw_scores"] is not None
    rank = client.post("/v1/rank", json=REQUEST).json()
    assert rank["ranking"][0]["choice"] == body["choice"]
    assert rank["decision"]["model"] == body["model"]


def test_batch_and_independent_booleans(client):
    results = client.post("/v1/decide/batch", json={"requests": [REQUEST, REQUEST]}).json()
    assert len(results) == 2
    assert results[0]["probabilities"] == results[1]["probabilities"]
    boolean = client.post("/v1/boolean", json={"state": "", "question": "Ready?"})
    assert boolean.status_code == 200
    assert isinstance(boolean.json()["value"], bool)
    labels = ["Ready?", "Needs review?"]
    response = client.post("/v1/multi-label", json={"state": "", "labels": labels})
    assert response.status_code == 200
    for label in labels:
        assert sum(response.json()[label]["decision"]["probabilities"].values()) == pytest.approx(1)


def test_threshold_abstention(client):
    result = client.post("/v1/decide", json={**REQUEST, "abstain_threshold": 1}).json()
    assert result["abstained"] is True
    assert result["choice"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {**REQUEST, "choices": ["billing"]},
        {**REQUEST, "choices": ["billing", "billing"]},
        {**REQUEST, "abstain_threshold": 1.1},
        {**REQUEST, "unexpected": "value"},
    ],
)
def test_invalid_decision_rejected(client, payload):
    assert client.post("/v1/decide", json=payload).status_code == 422


def test_other_input_bounds(client):
    assert client.post("/v1/decide/batch", json={"requests": []}).status_code == 422
    assert client.post("/v1/decide/batch", json={"requests": [REQUEST] * 65}).status_code == 422
    assert client.post("/v1/boolean", json={"state": "", "question": " "}).status_code == 422
    assert (
        client.post("/v1/multi-label", json={"state": "", "labels": ["a", "a"]}).status_code == 422
    )
    assert client.post("/v1/decide", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413


def test_bounded_concurrency_and_responsive_health():
    entered, release = Event(), Event()
    result = DecisionModel("demo").choose(**REQUEST)

    class SlowEngine:
        def choose(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return result

    with TestClient(create_app("demo", decision_model=SlowEngine(), max_concurrency=1)) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(client.post, "/v1/decide", json=REQUEST)
            try:
                assert entered.wait(5)
                assert client.get("/health").status_code == 200
                second = client.post("/v1/decide", json=REQUEST)
                assert second.status_code == 429
                assert second.headers["retry-after"] == "1"
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200


def test_backend_errors_have_actionable_response(client, monkeypatch):
    from opendecision.errors import ModelNotAvailableError

    def fail(**kwargs):
        raise ModelNotAvailableError("Run opendecision pull base first")

    monkeypatch.setattr(client.app.state.engine, "choose", fail)
    response = client.post("/v1/decide", json=REQUEST)
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "ModelNotAvailableError"


def test_openapi_and_models_contract(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/decide/batch" in schema["paths"]
    models = client.get("/v1/models").json()
    assert models["active_model"] == "demo"
    assert any(model["name"] == "base" for model in models["models"])

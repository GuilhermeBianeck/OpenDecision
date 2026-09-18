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


def test_statement_engine_contract_and_unsupported_threshold():
    from test_core import StatementBackend

    engine = DecisionModel(backend=StatementBackend([1.0, 3.0, -1.0]))
    with TestClient(create_app("demo", decision_model=engine)) as client:
        assert client.get("/health").json()["supports_statements"] is True
        body = client.post(
            "/v1/boolean", json={"state": "s", "question": "The charge was refunded."}
        ).json()
        assert body["method"] == "statement"
        assert body["value"] is True
        assert 0 < body["unsupported"] < 1
        abstained = client.post(
            "/v1/boolean",
            json={"state": "s", "question": "claim", "unsupported_threshold": 0.1},
        ).json()
        assert abstained["value"] is None and abstained["decision"]["abstained"]
        labels = client.post(
            "/v1/multi-label",
            json={"state": "s", "labels": ["a", "b"], "unsupported_threshold": 0.1},
        ).json()
        assert all(label["method"] == "statement" for label in labels.values())


def test_unsupported_threshold_is_rejected_without_statement_scoring(client):
    assert client.get("/health").json()["supports_statements"] is False
    response = client.post(
        "/v1/boolean", json={"state": "", "question": "Ready?", "unsupported_threshold": 0.5}
    )
    assert response.status_code == 422
    assert "statement scoring" in response.json()["detail"]
    plain = client.post("/v1/boolean", json={"state": "", "question": "Ready?"}).json()
    assert plain["method"] == "binary_choice" and plain["unsupported"] is None


def test_structured_state_and_options_over_http(client):
    body = client.post(
        "/v1/decide",
        json={
            "state": {"message": "billing support needed", "channel": "email"},
            "question": "Which team?",
            "choices": [
                {"label": "billing", "description": "invoices and payments"},
                "technical",
            ],
        },
    ).json()
    assert body["choice"] == "billing"
    assert set(body["probabilities"]) == {"billing", "technical"}
    assert (
        client.post(
            "/v1/boolean", json={"state": ["billing", "support"], "question": "billing?"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/multi-label", json={"state": {"a": "x" * 262_200}, "labels": ["x"]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/decide",
            json={"state": "", "question": "q", "choices": ["a", {"label": "a"}]},
        ).status_code
        == 422
    )


def test_ask_contract(client):
    body = client.post(
        "/v1/ask",
        json={
            "state": "billing support needed now",
            "questions": {
                "team": {"type": "choice", "question": "Team?", "choices": ["billing", "other"]},
                "ready": {"type": "boolean", "statement": "support needed"},
                "urgency": {
                    "type": "score",
                    "question": "Urgency?",
                    "levels": ["later", "now"],
                },
            },
        },
    )
    assert body.status_code == 200
    answers = body.json()
    assert list(answers) == ["team", "ready", "urgency"]
    assert answers["team"]["type"] == "choice" and answers["team"]["choice"] == "billing"
    assert answers["ready"]["type"] == "boolean" and answers["ready"]["method"] == "binary_choice"
    assert answers["urgency"]["type"] == "score" and answers["urgency"]["level"] == 1
    assert client.post("/v1/ask", json={"state": "", "questions": {}}).status_code == 422
    assert "/v1/ask" in client.get("/openapi.json").json()["paths"]
    many = {"state": "", "labels": [f"label {i}" for i in range(128)]}
    assert client.post("/v1/multi-label", json=many).status_code == 200
    many["labels"].append("one more")
    assert client.post("/v1/multi-label", json=many).status_code == 422


def test_score_contract(client):
    payload = {
        "state": "Billing needs billing support urgently.",
        "question": "How urgent is this?",
        "levels": ["not urgent", "somewhat urgent", "billing urgent"],
    }
    body = client.post("/v1/score", json=payload).json()
    assert body["legend"] == {"0": "not urgent", "1": "somewhat urgent", "2": "billing urgent"}
    assert sum(body["probabilities"].values()) == pytest.approx(1)
    assert body["level"] == 2
    assert 0 <= body["score"] <= 2
    assert body["decision"]["choice"] == "billing urgent"
    assert client.post("/v1/score", json={**payload, "levels": ["one"]}).status_code == 422
    assert "/v1/score" in client.get("/openapi.json").json()["paths"]


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

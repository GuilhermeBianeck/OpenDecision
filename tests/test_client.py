"""The Python HTTP client against a real loopback server, and the in-process path."""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("uvicorn")
import uvicorn
from test_systemone import SPEC_REQUEST

from opendecision import DecisionModel
from opendecision.client import COMPRESS_ABOVE_BYTES, Client, OpenDecisionHTTPError
from opendecision.server import create_app


@pytest.fixture(scope="module")
def client():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app("demo"), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    try:
        yield Client(f"http://127.0.0.1:{port}/")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_systemone_round_trip(client):
    body = client.systemone(SPEC_REQUEST)
    assert list(body["answers"]) == ["department", "frustration", "is_urgent"]
    assert body["usage"]["output_tokens"] == 0


def test_large_requests_are_sent_compressed(client):
    request = {**SPEC_REQUEST, "state": "word " * (COMPRESS_ABOVE_BYTES // 4)}
    assert client.systemone(request)["answers"]["department"]["type"] == "choice"


def test_native_ask_and_health(client):
    assert client.health()["status"] == "ok"
    answers = client.ask(
        state="s", questions={"q": {"type": "boolean", "statement": "It is true."}}
    )
    assert answers["q"]["type"] == "boolean"


def test_errors_keep_the_server_detail(client):
    with pytest.raises(OpenDecisionHTTPError) as raised:
        client.systemone({"state": "s"})
    assert raised.value.status == 422
    assert isinstance(raised.value.body["detail"], list)


@pytest.mark.parametrize("kwargs", [{"base_url": "ftp://x"}, {"timeout": 0}])
def test_bad_client_options(kwargs):
    with pytest.raises(ValueError):
        Client(**kwargs)


def test_in_process_matches_the_server(client):
    local = DecisionModel("demo").systemone(SPEC_REQUEST).model_dump()
    assert local == client.systemone(SPEC_REQUEST)

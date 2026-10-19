"""Limits, status codes and compressed bodies on /v1/systemone."""

from __future__ import annotations

import copy
import gzip
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_systemone import SPEC_REQUEST

from opendecision.server import create_app
from opendecision.server.app import SYSTEMONE_MAX_BODY_BYTES
from opendecision.server.body import MAX_BODY_BYTES, MAX_JSON_DEPTH

JSON = {"content-type": "application/json"}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app("demo")) as value:
        yield value


def post(client, body: bytes, **headers):
    return client.post("/v1/systemone", content=body, headers={**JSON, **headers})


def encoded(request=SPEC_REQUEST) -> bytes:
    return json.dumps(request).encode()


def with_choices(n: int) -> dict:
    request = copy.deepcopy(SPEC_REQUEST)
    request["questions"]["department"]["criteria"] = {f"option {i}": None for i in range(n)}
    return request


def with_levels(n: int) -> dict:
    request = copy.deepcopy(SPEC_REQUEST)
    request["questions"]["frustration"]["criteria"] = [f"level {i}" for i in range(n)]
    return request


def nested_state(depth: int) -> dict:
    """A request whose deepest container sits ``depth`` levels down."""
    state: object = {"leaf": "value"}
    for _ in range(depth - 2):  # the request object itself is level 1
        state = {"inner": state}
    return {**SPEC_REQUEST, "state": state}


# Limits ---------------------------------------------------------------------


def test_up_to_255_choice_options(client):
    assert post(client, encoded(with_choices(255))).status_code == 200
    assert post(client, encoded(with_choices(256))).status_code == 422


def test_two_to_ten_score_levels(client):
    assert post(client, encoded(with_levels(10))).status_code == 200
    assert post(client, encoded(with_levels(11))).status_code == 422
    assert post(client, encoded(with_levels(1))).status_code == 422


def test_nesting_is_capped(client):
    assert post(client, encoded(nested_state(MAX_JSON_DEPTH))).status_code == 200
    response = post(client, encoded(nested_state(MAX_JSON_DEPTH + 1)))
    assert response.status_code == 422
    assert "nesting" in response.json()["detail"]


def test_numbers_must_be_finite(client):
    body = encoded().replace(b'"state": "', b'"extra_number": 1e999, "state": "', 1)
    response = post(client, body)
    assert response.status_code == 422
    assert "finite" in response.json()["detail"]


def test_body_cap_is_32_mib_here_and_1_mib_elsewhere(client):
    # Whitespace is valid JSON, so padding grows the body without changing it.
    padded = encoded() + b" " * (2 * MAX_BODY_BYTES)
    assert post(client, padded).status_code == 200
    assert client.post("/v1/decide", content=padded, headers=JSON).status_code == 413
    too_big = encoded() + b" " * SYSTEMONE_MAX_BODY_BYTES
    response = post(client, too_big)
    assert response.status_code == 413
    assert "32 MiB" in response.json()["detail"]


# Status codes ---------------------------------------------------------------


@pytest.mark.parametrize("body", [b"{", b"not json", b'{"state": NaN}', b"\xff\xfe"])
def test_400_for_bodies_that_are_not_json(client, body):
    assert post(client, body).status_code == 400


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", ""])
def test_415_for_bodies_that_are_not_declared_json(client, content_type):
    response = client.post(
        "/v1/systemone", content=encoded(), headers={"content-type": content_type}
    )
    assert response.status_code == 415


def test_json_media_type_parameters_and_suffixes_are_accepted(client):
    for content_type in ["application/json; charset=utf-8", "application/vnd.api+json"]:
        response = client.post(
            "/v1/systemone", content=encoded(), headers={"content-type": content_type}
        )
        assert response.status_code == 200


def test_422_for_valid_json_that_is_not_a_request(client):
    assert post(client, b'{"state": "s"}').status_code == 422


# Compression ----------------------------------------------------------------


@pytest.mark.parametrize("coding", ["gzip", "x-gzip", "GZIP"])
def test_gzip_bodies(client, coding):
    response = post(client, gzip.compress(encoded()), **{"content-encoding": coding})
    assert response.status_code == 200


def test_zstd_bodies(client):
    try:
        from compression import zstd

        compressed = zstd.compress(encoded())
    except ImportError:
        zstandard = pytest.importorskip("zstandard")
        compressed = zstandard.ZstdCompressor().compress(encoded())
    assert post(client, compressed, **{"content-encoding": "zstd"}).status_code == 200


def test_identity_encoding_is_a_no_op(client):
    assert post(client, encoded(), **{"content-encoding": "identity"}).status_code == 200


@pytest.mark.parametrize("coding", ["br", "deflate", "gzip, gzip", "gzip, zstd"])
def test_415_for_unknown_or_stacked_encodings(client, coding):
    assert post(client, encoded(), **{"content-encoding": coding}).status_code == 415


def test_400_for_corrupt_or_truncated_gzip(client):
    assert post(client, b"not gzip", **{"content-encoding": "gzip"}).status_code == 400
    truncated = gzip.compress(encoded())[:-12]
    assert post(client, truncated, **{"content-encoding": "gzip"}).status_code == 400


def test_413_when_a_small_body_inflates_past_the_cap(client):
    bomb = gzip.compress(b" " * (SYSTEMONE_MAX_BODY_BYTES + 1))
    assert len(bomb) < MAX_BODY_BYTES
    response = post(client, bomb, **{"content-encoding": "gzip"})
    assert response.status_code == 413
    assert "Decompressed" in response.json()["detail"]

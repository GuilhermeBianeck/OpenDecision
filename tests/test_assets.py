"""Images sent as the shared format's `assets`: references, placeholders, uploads."""

from __future__ import annotations

import base64
import json

import pytest
from test_box import Seeing
from test_images import JPEG, PNG

from opendecision import DecisionModel
from opendecision.images import resolve_assets

PNG_B64 = base64.b64encode(PNG).decode()
JPG_B64 = base64.b64encode(JPEG).decode()


def request(**extra):
    base = {
        "state": {"photo": "ref-found", "note": "seen near ref-found's kennel"},
        "model": "m",
        "assets": {
            "ref-found": {"type": "image", "format": "png", "data_base64": PNG_B64},
            "ref-poster": {"format": "jpg", "data_base64": JPG_B64},
        },
        "questions": {
            "same": {"type": "noul", "instructions": "Is it the dog in this picture: ref-poster?"},
            "size": {"type": "score", "criteria": ["small", "ref-poster sized"]},
            "pick": {"type": "choice", "criteria": {"ref-found": "keep the key", "other": None}},
        },
    }
    base.update(extra)
    return base


def test_references_become_numbered_placeholders_and_keys_stay_text():
    out = resolve_assets(request())
    assert "assets" not in out
    assert out["state"] == {"photo": "[image 1]", "note": "seen near [image 2]'s kennel"}
    assert out["questions"]["same"]["instructions"] == "Is it the dog in this picture: [image 3]?"
    assert out["questions"]["size"]["criteria"] == ["small", "[image 4] sized"]
    assert list(out["questions"]["pick"]["criteria"]) == ["ref-found", "other"]  # keys untouched
    assert len(out["images"]) == 4  # every use adds a copy
    assert out["images"][2].startswith("data:image/jpeg;base64,")
    assert out["image_references"]["[image 3]"] == "ref-poster"


def test_longer_references_win_and_matches_do_not_overlap():
    assets = {
        "cat": {"format": "png", "data_base64": PNG_B64},
        "catalog": {"format": "png", "data_base64": PNG_B64},
    }
    out = resolve_assets({"state": "catalog then cat", "questions": {}, "assets": assets})
    assert out["state"] == "[image 1] then [image 2]"
    assert [out["image_references"][p] for p in ("[image 1]", "[image 2]")] == ["catalog", "cat"]


@pytest.mark.parametrize(
    ("assets", "message"),
    [
        ({"unused": {"format": "png", "data_base64": PNG_B64}}, "not referenced"),
        ({"ref-found": {"data_base64": PNG_B64}}, "format"),
        ({"ref-found": {"format": "gif", "data_base64": PNG_B64}}, "format"),
        ({"ref-found": {"format": "png", "data_base64": "%%%"}}, "base64"),
        (
            {"ref-found": {"format": "png", "data_base64": base64.b64encode(b"text").decode()}},
            "PNG",
        ),
        ({"ref-found": {"part": "missing"}}, "part"),
        ({"ref-found": {"type": "audio", "format": "png", "data_base64": PNG_B64}}, "image"),
        ({" ": {"format": "png", "data_base64": PNG_B64}}, "empty"),
    ],
)
def test_bad_assets_are_rejected(assets, message):
    with pytest.raises(ValueError, match=message):
        resolve_assets({"state": "ref-found", "questions": {}, "assets": assets})


def test_score_legends_repeat_the_reference_not_the_placeholder():
    engine = DecisionModel(backend=Seeing())
    body = engine.systemone(request()).model_dump()
    assert body["answers"]["size"]["legend"] == {"0": "small", "1": "ref-poster sized"}


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from opendecision.server import create_app

    with TestClient(create_app("demo", decision_model=DecisionModel(backend=Seeing()))) as value:
        yield value


def test_json_assets_over_http(client):
    response = client.post("/v1/systemone", json=request())
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["size"]["legend"]["1"] == "ref-poster sized"


def test_asset_errors_are_400_over_http(client):
    bad = request(assets={"never-used": {"format": "png", "data_base64": PNG_B64}})
    response = client.post("/v1/systemone", json=bad)
    assert response.status_code == 400
    assert "not referenced" in response.json()["detail"]


def test_multipart_uploads_name_their_parts(client):
    payload = request(
        assets={"ref-found": {"type": "image", "part": "found"}, "ref-poster": {"part": "poster"}}
    )
    response = client.post(
        "/v1/systemone",
        files={
            "request": (None, json.dumps(payload), "application/json"),
            "found": ("found.png", PNG, "image/png"),
            "poster": ("poster.jpg", JPEG, "image/jpeg"),
        },
    )
    assert response.status_code == 200, response.text
    assert set(response.json()["answers"]) == {"same", "size", "pick"}


def test_multipart_without_a_request_part_is_400(client):
    response = client.post("/v1/systemone", files={"found": ("found.png", PNG, "image/png")})
    assert response.status_code == 400

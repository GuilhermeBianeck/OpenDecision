"""Images in requests: accepted forms, placeholders, limits and refusals."""

from __future__ import annotations

import base64

import pytest

from opendecision import DecisionModel
from opendecision.images import MAX_IMAGES, extract_images, image_type, to_data_uri
from opendecision.schemas import DecisionRequest

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
PNG_B64 = base64.b64encode(PNG).decode()
PNG_URI = f"data:image/png;base64,{PNG_B64}"


def test_signatures_decide_the_type_not_the_claim():
    assert image_type(PNG) == "image/png"
    assert image_type(JPEG) == "image/jpeg"
    assert image_type(b"plain text") is None
    # A JPEG labelled as PNG is still a JPEG.
    lying = "data:image/png;base64," + base64.b64encode(JPEG).decode()
    _, images = extract_images("s", [lying])
    assert images[0].startswith("data:image/jpeg;base64,")


def test_images_inside_state_become_placeholders():
    state = {
        "note": "see attached",
        "photo": PNG_URI,
        "parts": [{"type": "image_url", "image_url": {"url": PNG_URI}}, "text stays"],
        "raw": PNG_B64,  # bare base64 of image bytes, one unbroken token
    }
    lifted, images = extract_images(state)
    assert lifted == {
        "note": "see attached",
        "photo": "[image 1]",
        "parts": ["[image 2]", "text stays"],
        "raw": "[image 3]",
    }
    assert all(uri.startswith("data:image/png;base64,") for uri in images)


def test_top_level_images_are_numbered_first():
    lifted, images = extract_images(f"before {PNG_URI}", [PNG_URI])
    assert len(images) == 1  # text containing a URI mid-sentence is not an image
    lifted, images = extract_images([PNG_URI], [{"type": "base64", "data": PNG_B64}])
    assert lifted == ["[image 2]"] and len(images) == 2


def test_ordinary_text_and_links_inside_state_are_left_alone():
    state = {"link": "https://example.com/a.png", "word": "iVBORw0KGgo"}
    assert extract_images(state) == (state, [])


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (["https://example.com/a.png"], "not fetched"),
        (["not base64 at all!"], "data URI or base64"),
        ([base64.b64encode(b"GIF? no").decode()], "data URI or base64"),
        ([PNG_URI] * (MAX_IMAGES + 1), f"at most {MAX_IMAGES}"),
    ],
)
def test_bad_images_are_rejected(images, message):
    with pytest.raises(ValueError, match=message):
        extract_images("s", images)


def test_oversized_images_are_rejected():
    from opendecision import images

    with pytest.raises(ValueError, match="byte limit"):
        to_data_uri(PNG + b"\x00" * images.MAX_IMAGE_BYTES)


def test_requests_carry_validated_images():
    request = DecisionRequest(state="s", question="q", choices=["a", "b"], images=[PNG_URI])
    assert request.images == [PNG_URI]
    with pytest.raises(ValueError):
        DecisionRequest(state="s", question="q", choices=["a", "b"], images=["data:x"])


def test_a_text_only_backend_refuses_images_clearly():
    with pytest.raises(ValueError, match="cannot read images.*lightning_vision"):
        DecisionModel("demo").choose(state="s", question="q", choices=["a", "b"], images=[PNG_URI])


def test_systemone_lifts_images_and_refuses_them_on_text_backends():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from opendecision.server import create_app
    from opendecision.systemone import SystemOneRequest

    request = SystemOneRequest.model_validate(
        {
            "state": {"receipt": PNG_URI, "note": "lunch"},
            "model": "m",
            "questions": {"q": {"type": "noul", "instructions": "It is a receipt."}},
        }
    )
    assert request.state == {"receipt": "[image 1]", "note": "lunch"}
    assert request.images == [PNG_URI]
    with TestClient(create_app("demo")) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "state": "s",
                "model": "m",
                "images": [PNG_URI],
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            },
        )
    assert response.status_code == 422
    assert "lightning_vision" in response.text

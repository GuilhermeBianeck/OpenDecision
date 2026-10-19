"""The experimental box question: parsing, the native call and the shared format."""

from __future__ import annotations

import pytest
from test_core import StatementBackend
from test_images import PNG_URI

from opendecision import DecisionModel
from opendecision.backends.vision_mlx import parse_box


@pytest.mark.parametrize(
    ("text", "box"),
    [
        ('{"bbox_2d": [619, 205, 815, 465], "label": "red circle"}', [0.619, 0.205, 0.815, 0.465]),
        ("[815, 465, 619, 205]", [0.619, 0.205, 0.815, 0.465]),
        ("[900, 50, 100, 1200]", [0.1, 0.05, 0.9, 1.0]),
        ("I cannot see one.", None),
        ("[5, 5, 5, 9]", None),
        ("[1, 2, 3]", None),
    ],
)
def test_boxes_are_parsed_ordered_and_clamped(text, box):
    assert parse_box(text) == (pytest.approx(box) if box else None)


class Seeing(StatementBackend):
    supports_images = True

    def __init__(self, box=(0.5, 0.25, 0.75, 0.5)):
        super().__init__()
        self.box = list(box) if box else None
        self.located = []

    def locate(self, state, target, image):
        self.located.append((state, target))
        return self.box, (640, 480), 17


def test_locate_reports_fractions_and_pixels():
    backend = Seeing()
    result = DecisionModel(backend=backend).locate(state="a photo", target="the cup", image=PNG_URI)
    assert result.found and result.box == [0.5, 0.25, 0.75, 0.5]
    assert result.pixels == [320, 120, 480, 240]
    assert result.output_tokens == 17
    assert backend.located == [("a photo", "the cup")]


def test_locate_reports_a_missing_box_as_not_found():
    result = DecisionModel(backend=Seeing(box=None)).locate(state="s", target="x", image=PNG_URI)
    assert not result.found and result.box is None and result.pixels is None


def test_text_backends_cannot_locate():
    with pytest.raises(ValueError, match="cannot locate.*lightning_vision"):
        DecisionModel("demo").locate(state="s", target="x", image=PNG_URI)


BOX_REQUEST = {
    "state": {"photo": PNG_URI, "note": "kitchen"},
    "model": "m",
    "questions": {
        "where": {"type": "box", "instructions": "the cup"},
        "is_kitchen": {"type": "noul", "instructions": "The photo is of a kitchen."},
    },
}


def test_box_questions_in_the_shared_format():
    engine = DecisionModel(backend=Seeing())
    body = engine.systemone(BOX_REQUEST).model_dump()
    assert list(body["answers"]) == ["where", "is_kitchen"]
    assert body["answers"]["where"] == {
        "type": "box",
        "box": [0.5, 0.25, 0.75, 0.5],
        "pixels": [320, 120, 480, 240],
    }
    assert body["answers"]["is_kitchen"]["type"] == "noul"
    assert body["usage"]["output_tokens"] == 17


def test_box_questions_need_exactly_one_image():
    engine = DecisionModel(backend=Seeing())
    with pytest.raises(ValueError, match="exactly one image"):
        engine.systemone({**BOX_REQUEST, "state": "no image here"})
    with pytest.raises(ValueError, match="exactly one image"):
        engine.systemone({**BOX_REQUEST, "images": [PNG_URI]})

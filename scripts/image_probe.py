"""Measure image-state decisions on generated pictures with known answers.

Each image draws one to four shapes of a single kind and colour on a white
canvas at seeded positions. Four questions are asked about it, one per question
type: the shape (choice), the colour (choice), the count (score over "one" to
"four") and a true/false statement about the count. The answers are known by
construction, so no labelling is involved. Results are saved after every
image, so a run stopped for memory keeps what finished.

    python scripts/image_probe.py --model lightning_vision \\
        --output benchmarks/reports/vision/lightning_vision.json
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import random
import statistics
import time
from pathlib import Path

from opendecision import DecisionModel

SHAPES = ("circle", "square", "triangle")
COLOURS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 160, 70)}
COUNTS = ("one", "two", "three", "four")


def draw(shape: str, colour: str, count: int, seed: int) -> str:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    image = Image.new("RGB", (640, 480), "white")
    canvas = ImageDraw.Draw(image)
    cells = rng.sample([(x, y) for x in range(4) for y in range(3)], count)
    for x, y in cells:
        left, top, size = 30 + x * 150, 30 + y * 150, 100
        box = (left, top, left + size, top + size)
        if shape == "circle":
            canvas.ellipse(box, fill=COLOURS[colour])
        elif shape == "square":
            canvas.rectangle(box, fill=COLOURS[colour])
        else:
            canvas.polygon(
                [(left + size / 2, top), (left, top + size), (left + size, top + size)],
                fill=COLOURS[colour],
            )
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="lightning_vision")
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    combos = list(itertools.product(SHAPES, COLOURS, range(1, 5)))
    rng.shuffle(combos)
    model = DecisionModel(args.model)
    rows = []
    for index, (shape, colour, count) in enumerate(combos[: args.images]):
        uri = draw(shape, colour, count, seed=args.seed * 1000 + index)
        threshold = rng.randint(1, 3)
        questions = {
            "shape": {
                "type": "choice",
                "question": "What shape are the objects in the image?",
                "choices": list(SHAPES),
            },
            "colour": {
                "type": "choice",
                "question": "What colour are the objects in the image?",
                "choices": list(COLOURS),
            },
            "count": {
                "type": "score",
                "question": "How many objects are in the image?",
                "levels": list(COUNTS),
            },
            "more": {
                "type": "boolean",
                "statement": f"The image shows more than {COUNTS[threshold - 1]} objects.",
            },
        }
        started = time.perf_counter()
        answers = model.ask(state="[image 1]", questions=questions, images=[uri])
        latency = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "shape": shape,
                "colour": colour,
                "count": count,
                "latency_ms": latency,
                "correct": {
                    "shape": answers["shape"].choice == shape,
                    "colour": answers["colour"].choice == colour,
                    "count": answers["count"].level == count - 1,
                    "more": answers["more"].value == (count > threshold),
                },
            }
        )
        report = {
            "model": args.model,
            "images": len(rows),
            "accuracy": {
                key: sum(row["correct"][key] for row in rows) / len(rows)
                for key in ("shape", "colour", "count", "more")
            },
            "latency_ms_per_image": {"p50": statistics.median(r["latency_ms"] for r in rows)},
            "rows": rows,
        }
        try:
            import mlx.core as mx

            report["peak_memory_bytes"] = mx.get_peak_memory()
        except ImportError:
            pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

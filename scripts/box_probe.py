"""Measure the experimental box question on generated scenes with known boxes.

Each scene places one target (a shape in a colour) and two distractors of
other shapes or colours at seeded, non-overlapping positions and sizes. The
model is asked to locate the target; the answer is scored by intersection over
union with the true box. Results are saved after every scene.

    python scripts/box_probe.py --output benchmarks/reports/vision/lightning_vision-box.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
from pathlib import Path

from opendecision import DecisionModel

SHAPES = ("circle", "square", "triangle")
COLOURS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 160, 70)}
WIDTH, HEIGHT = 640, 480


def scene(seed: int) -> tuple[str, str, list[int]]:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    kinds = rng.sample([(s, c) for s in SHAPES for c in COLOURS], 3)
    cells = rng.sample([(x, y) for x in range(3) for y in range(2)], 3)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas = ImageDraw.Draw(image)
    target_box: list[int] = []
    for index, ((shape, colour), (cx, cy)) in enumerate(zip(kinds, cells)):
        size = rng.randint(70, 150)
        left = cx * 213 + rng.randint(5, 213 - size - 5)
        top = cy * 240 + rng.randint(5, 240 - size - 5)
        box = [left, top, left + size, top + size]
        if shape == "circle":
            canvas.ellipse(box, fill=COLOURS[colour])
        elif shape == "square":
            canvas.rectangle(box, fill=COLOURS[colour])
        else:
            canvas.polygon(
                [(left + size / 2, top), (left, top + size), (left + size, top + size)],
                fill=COLOURS[colour],
            )
        if index == 0:
            target_box = box
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    shape, colour = kinds[0]
    return uri, f"the {colour} {shape}", target_box


def iou(a: list[int], b: list[int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="lightning_vision")
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume", action="store_true", help="Keep the scenes already in --output and continue"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    if args.resume and args.output.exists():
        rows = json.loads(args.output.read_text())["rows"]
    model = DecisionModel(args.model)
    for index in range(len(rows), args.scenes):
        uri, target, truth = scene(args.seed * 1000 + index)
        result = model.locate(state="[image 1]", target=target, image=uri)
        rows.append(
            {
                "target": target,
                "truth": truth,
                "pixels": result.pixels,
                "iou": iou(result.pixels, truth) if result.pixels else 0.0,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
            }
        )
        scores = [row["iou"] for row in rows]
        report = {
            "model": args.model,
            "scenes": len(rows),
            "found": sum(row["pixels"] is not None for row in rows) / len(rows),
            "mean_iou": statistics.mean(scores),
            "hit_rate_iou_0_5": sum(score >= 0.5 for score in scores) / len(rows),
            "latency_ms": {"p50": statistics.median(row["latency_ms"] for row in rows)},
            "rows": rows,
        }
        try:
            import mlx.core as mx

            report["peak_memory_bytes"] = mx.get_peak_memory()
        except ImportError:
            pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        saved = json.loads(args.output.read_text())
        print(json.dumps({k: v for k, v in saved.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

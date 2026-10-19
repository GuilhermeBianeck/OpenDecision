# Images (experimental)

`lightning_vision` reads images in the state. It is the `lightning` checkpoint,
which ships its own vision tower, loaded once through mlx-vlm on Apple silicon:

```bash
python -m pip install -e '.[vision]'
opendecision pull lightning_vision --device mps   # same files as lightning
opendecision serve --model lightning_vision
```

It answers every question type with `lightning`'s prompt contract, so text-only
answers match `lightning` to within 0.04 in log-probability. Use `lightning`
for text: it caches a shared state across questions, and this backend encodes
every prompt in full.

## Sending images

Up to four PNG, JPEG, GIF or WebP images per request, 20 MiB each, recognised by
their bytes. In the shared format they may sit anywhere in `state`, as a
`data:image/...;base64,` URI, as bare base64, or as an image part such as
`{"type": "image_url", "image_url": {"url": "data:..."}}`, or in a top-level
`images` list. Each is lifted out and replaced by `[image N]`, so the state the
model reads stays text:

```json
{
  "state": {"receipt": "data:image/png;base64,iVBORw0...", "submitted_by": "Ana"},
  "model": "any",
  "questions": {
    "category": {"type": "choice", "instructions": "What kind of expense is this?",
                 "criteria": {"meals": null, "travel": null, "equipment": null}},
    "over_limit": {"type": "noul", "instructions": "The total is above 50 euros."},
    "total": {"type": "box", "instructions": "the total amount"}
  }
}
```

In Python, pass data URIs: `model.ask(state=..., questions=..., images=[uri])`
or `model.choose(..., images=[uri])`. Image URLs are refused rather than
fetched, because inference makes no network requests; send the bytes. A text
backend given images answers 422 and names the vision backend. Images are
downscaled to one megapixel, about 1,024 image tokens each.

## Box questions

A `box` question, or `DecisionModel.locate(state=..., target=..., image=uri)`,
returns where the named thing is in the request's single image:
`[left, top, right, bottom]` as fractions of the image and as pixels of the
original, or `null` when the model gives no usable box. This is the one
question type that generates text, so `usage.output_tokens` is no longer zero
when a request has one.

## Measured

On generated pictures with known answers (`scripts/image_probe.py`,
`scripts/box_probe.py`; reports in
[`benchmarks/reports/vision/`](../benchmarks/reports/vision/)):

| Probe | Result |
| --- | --- |
| Shape, 24 images (choice) | 100 % |
| Colour, 24 images (choice) | 100 % |
| Count from one to four (score) | 95.8 % |
| "More than N objects" (statement) | 95.8 % |
| Locating one target among two distractors (box), 6 scenes | all found, mean IoU 0.92 |

Four questions about one image take 2.1 s; one box takes 1.4 s. Peak memory is
4.02 GB. The box probe is a small sample: on the 16 GB reference machine,
with other applications running, the memory watchdog that halts any job
leaving under 35 % free stopped it after six scenes. These are synthetic
shapes, not photographs or documents; measure on your own images before
relying on either question.

"""Images in the state, for the decision-trained checkpoint's vision build.

Qwen3.5 places image tokens with three-dimensional rotary positions, which only
the mlx-vlm language model implements, so this backend loads the checkpoint
once through mlx-vlm and serves text and images alike with the same prompt
contract as ``lightning``. Text-only answers match the mlx-lm path to within
0.04 in log-probability; the text-only default stays on mlx-lm, which caches a
shared state across questions and this backend does not.

Images are downscaled to at most one megapixel, about 1,024 image tokens each.
"""

from __future__ import annotations

import io
import math
import re
import time
from typing import Any

from opendecision.backends.catalog import MODEL_SPECS
from opendecision.backends.decision_mlx import NO_UNKNOWN, STATEMENT_OPTIONS, DecisionMLXBackend
from opendecision.backends.decoder import LETTERS
from opendecision.errors import BackendError
from opendecision.images import decode_data_uri
from opendecision.memory import MLX_CACHE_LIMIT_BYTES, check_memory

MAX_IMAGE_PIXELS = 1 << 20


def _pil(uri: str) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(decode_data_uri(uri)))
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / pixels)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


class VisionDecisionBackend(DecisionMLXBackend):
    supports_images = True

    def __init__(self, *, permutations: int = 1, **kwargs: Any):
        if permutations != 1:
            raise BackendError("Order averaging is not available on the vision backend.")
        super().__init__(**kwargs)
        self._processor: Any = None
        self._prepare: Any = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "runtime": "mlx-vlm",
            "cache_strategy": "none; every prompt is encoded in full",
            "images": f"up to 4 per request, downscaled to {MAX_IMAGE_PIXELS} pixels",
        }

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                import mlx.core as mx
                from huggingface_hub import snapshot_download
                from mlx_vlm import load
                from mlx_vlm.utils import prepare_inputs

                from opendecision.registry import DOWNLOAD_PATTERNS
            except ImportError as exc:
                raise BackendError(
                    "The vision backend needs mlx-vlm: pip install -e '.[vision]'"
                ) from exc
            try:
                if not mx.metal.is_available():
                    raise BackendError("MLX scoring requires an available Apple Metal GPU.")
                check_memory(MODEL_SPECS[self.name])
                mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
                path = snapshot_download(
                    self.model_id,
                    revision=self.revision,
                    local_files_only=True,
                    allow_patterns=DOWNLOAD_PATTERNS,
                )
                model, processor = load(path, trust_remote_code=False)
                quant = getattr(model.config, "quantization", None) or {}
                if quant and (quant.get("bits") != 4 or quant.get("group_size") != 64):
                    raise BackendError("Checkpoint does not match the pinned Q4/group64 format.")
                tokenizer = processor.tokenizer
                ids = [
                    tokenizer.encode(self.label_prefix + c, add_special_tokens=False)
                    for c in LETTERS
                ]
                if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != 26:
                    raise BackendError("Option letters must be distinct single tokens.")
                self._mx, self._prepare, self._processor = mx, prepare_inputs, processor
                self._tokenizer, self._letter_ids = tokenizer, [row[0] for row in ids]
                self._model = model
                self.load_time_ms = (time.perf_counter() - started) * 1000
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(
                    f"Cannot load cached {self.name}: {type(exc).__name__}: {exc}. "
                    f"Run 'opendecision pull {self.name}' first."
                ) from exc

    def _score(self, state: str, question: str, images: list[str], count: int) -> list[float]:
        mx = self._mx
        text = self._prompt_text(state, question, len(images))
        if len(self._encode(text)) > self.max_length:
            raise BackendError(
                f"Prompt exceeds max_length {self.max_length} before image tokens; "
                "shorten the state or raise max_length."
            )
        inputs = self._prepare(
            self._processor,
            images=[_pil(uri) for uri in images] or None,
            prompts=[text],
            image_token_index=self._model.config.image_token_index,
        )
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        output = self._model(inputs["input_ids"], mask=inputs.get("attention_mask"), **extra)
        logits = getattr(output, "logits", output)[0, -1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        selected = logprobs[mx.array(self._letter_ids[:count])]
        mx.eval(selected)
        return selected.tolist()

    def score_batch(self, requests: list[Any]) -> list[list[float]]:
        with self._lock:
            self.load()
            return self._guard(
                lambda: [
                    self._score(
                        r.state_text,
                        self._question_text(r.question, self._candidates(r)),
                        r.images,
                        len(r.choices),
                    )
                    for r in requests
                ]
            )

    def score_statements(self, requests: list[Any]) -> list[list[float]]:
        with self._lock:
            self.load()
            rows = self._guard(
                lambda: [
                    self._score(
                        r.state_text,
                        self._question_text(r.statement, list(STATEMENT_OPTIONS)),
                        r.images,
                        len(STATEMENT_OPTIONS),
                    )
                    for r in requests
                ]
            )
            return [[row[0], NO_UNKNOWN, row[1]] for row in rows]

    def locate(
        self, state: str, target: str, image: str
    ) -> tuple[list[float] | None, tuple[int, int], int]:
        """The box around ``target`` in ``image``, the image's size, and tokens written.

        The box is ``[left, top, right, bottom]`` as fractions of the image.

        Unlike every other question this one generates text: the model writes
        the box as JSON in Qwen's 0-1000 grid, which is parsed and normalised.
        ``None`` means the answer held no usable box.
        """
        from mlx_vlm import generate
        from PIL import Image

        size = Image.open(io.BytesIO(decode_data_uri(image))).size
        with self._lock:
            self.load()
            prompt = (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"record: {state}\nLocate {target}. Output its bounding box as JSON: "
                '{"bbox_2d": [x1, y1, x2, y2]}.<|im_end|>\n'
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            result = self._guard(
                lambda: generate(
                    self._model,
                    self._processor,
                    prompt,
                    image=[_pil(image)],
                    max_tokens=BOX_MAX_TOKENS,
                    temperature=0.0,
                    verbose=False,
                )
            )
            tokens = int(getattr(result, "generation_tokens", 0))
            return parse_box(getattr(result, "text", "")), size, tokens


BOX_MAX_TOKENS = 48


def parse_box(text: str) -> list[float] | None:
    """Four numbers on Qwen's 0-1000 grid as ordered, clamped fractions, or None."""
    number = r"\s*(-?\d+(?:\.\d+)?)\s*"
    match = re.search(r"\[" + ",".join([number] * 4) + r"\]", text)
    if match is None:
        return None
    numbers = [float(n) for n in match.groups()]
    x1, y1, x2, y2 = (min(max(n / 1000, 0.0), 1.0) for n in numbers)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]

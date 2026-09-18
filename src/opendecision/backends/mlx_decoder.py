"""Offline, quantized Apple-silicon option scoring with native chat templates.

Hybrid recurrent models cannot rewind their state like a pure KV cache. Each
question therefore forks an evaluated prefix cache; no state crosses requests.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from opendecision.backends.decoder import (
    LETTERS,
    MIN_STATE_TOKENS,
    SYSTEM_PROMPT,
    DecoderBackend,
)
from opendecision.errors import BackendError


def common_prefix_length(rows: list[list[int]]) -> int:
    """Keep at least one suffix token, even for a single or identical prompt."""
    limit = min(map(len, rows)) - 1
    for index in range(limit):
        if any(row[index] != rows[0][index] for row in rows[1:]):
            return index
    return max(0, limit)


class MLXDecoderBackend(DecoderBackend):
    def __init__(self, *, device: str = "auto", precision: str | None = None, **kwargs: Any):
        if device not in {"auto", "mps"}:
            raise BackendError("MLX checkpoints require Apple silicon: device='auto' or 'mps'.")
        if precision not in {None, "q4"}:
            raise BackendError("Pinned MLX checkpoints use precision='q4'.")
        super().__init__(device="mps", precision=None, **kwargs)
        self.precision = "q4"
        self._mx: Any = None
        self._lm: Any = None
        self._make_cache: Any = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "runtime": "mlx-lm",
            "prompt_format": "native-chat-template-v1",
            "thinking": False,
            "cache_strategy": "fork evaluated common prefix per question",
            "question_execution": "sequential suffixes; no tensor batch claim",
            "quantization": "affine-4bit-group64",
        }

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                import mlx.core as mx
                from huggingface_hub import snapshot_download
                from mlx_lm import load
                from mlx_lm.models.cache import make_prompt_cache

                if not mx.metal.is_available():
                    raise BackendError("MLX scoring requires an available Apple Metal GPU.")
                # Resolve locally first and pass a path: mlx-lm cannot implicitly fetch.
                path = snapshot_download(
                    self.model_id, revision=self.revision, local_files_only=True
                )
                model, tokenizer, config = load(
                    path, tokenizer_config={"trust_remote_code": False}, return_config=True
                )
                quant = config.get("quantization", {})
                if quant.get("bits") != 4 or quant.get("group_size") != 64:
                    raise BackendError("Checkpoint does not match the pinned Q4/group64 format.")
                ids = [tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
                if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != 26:
                    raise BackendError("Option letters must be distinct single tokens.")
                model.eval()
                self._lm = getattr(model, "language_model", model)
                self._mx, self._make_cache = mx, make_prompt_cache
                self._tokenizer, self._letter_ids = tokenizer, [row[0] for row in ids]
                self._model = model
                self.load_time_ms = (time.perf_counter() - started) * 1000
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(
                    f"Cannot load cached {self.name}: {type(exc).__name__}: {exc}. "
                    "Install opendecision[mlx], then explicitly pull the checkpoint."
                ) from exc

    @staticmethod
    def _question_text(question: str, options: list[str]) -> str:
        lines = [f"Question: {question}", "Options:"]
        lines.extend(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
        return "\n".join(lines) + "\nAnswer with the letter of the best option."

    def _prompt_ids(self, state: str, question: str) -> list[int]:
        messages = []
        if self.template != "short":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": f"Context:\n{state}\n\n{question}"})
        return list(self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        ))

    def _fit_prompts(self, state: str, questions: list[str]) -> list[list[int]]:
        state_ids = self._encode(state)
        keep = len(state_ids)
        minimum = min(MIN_STATE_TOKENS, keep)
        while True:
            text = state if keep == len(state_ids) else self._tokenizer.decode(state_ids[:keep])
            rows = [self._prompt_ids(text, q) for q in questions]
            overflow = max(map(len, rows)) - self.max_length
            if overflow <= 0:
                if keep < len(state_ids):
                    self._truncated_states += 1
                return rows
            keep -= max(1, overflow)
            if keep < minimum:
                raise BackendError("Question and options too long while preserving state.")

    def _last_logits(self, ids: list[int], cache: Any) -> Any:
        # Avoid projecting every input position to the complete vocabulary.
        hidden = self._lm.model(self._mx.array([ids]), cache=cache)[:, -1:, :]
        if hasattr(self._lm, "lm_head"):
            return self._lm.lm_head(hidden)[0, -1]
        return self._lm.model.embed_tokens.as_linear(hidden)[0, -1]

    def _run_group(self, state_text: str, suffixes: list[str]) -> list[list[float]]:
        mx = self._mx
        prompts = self._fit_prompts(state_text, suffixes)
        # A single question needs no fork or separate prefix invocation.
        shared = common_prefix_length(prompts) if len(prompts) > 1 else 0
        cache = self._make_cache(self._lm)
        if shared:
            hidden = self._lm.model(mx.array([prompts[0][:shared]]), cache=cache)
            mx.eval(hidden, *[c.state for c in cache])
        rows = []
        for prompt in prompts:
            fork = copy.deepcopy(cache) if shared else self._make_cache(self._lm)
            logits = self._last_logits(prompt[shared:], fork).astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits)
            selected = logprobs[mx.array(self._letter_ids)]
            mx.eval(selected)  # include completed GPU work in caller wall timings
            rows.append(selected.tolist())
        return rows

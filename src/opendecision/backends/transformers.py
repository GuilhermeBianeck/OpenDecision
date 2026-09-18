"""Offline, safely serialized Transformers inference for the registered models."""

from __future__ import annotations

import math
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

from opendecision.backends.catalog import MODEL_SPECS
from opendecision.errors import BackendError

if TYPE_CHECKING:
    from opendecision.schemas import DecisionRequest, StatementRequest


DEFAULT_MAX_LENGTH_CAP = 2048
"""Default sequence limit: the checkpoint's context, capped here for predictable memory."""


def default_max_length(context_limit: int) -> int:
    return min(context_limit, DEFAULT_MAX_LENGTH_CAP)


def select_device(requested: str, torch: Any) -> str:
    """Resolve ``auto`` to CUDA, then MPS, then CPU.

    MPS measured 1.2-3.5x faster than CPU for every registered model across
    32-1000 token states on the reference 16 GB Apple Silicon machine
    (see docs/model-licenses.md). Pass ``device="cpu"`` to opt out.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise BackendError("MPS is unavailable. Use device='cpu'.")
        return requested
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise BackendError("CUDA is unavailable. Use device='cpu'.")
        try:
            index = 0 if requested == "cuda" else int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise BackendError("CUDA device must be 'cuda' or 'cuda:<index>'.") from exc
        if not 0 <= index < torch.cuda.device_count():
            raise BackendError(f"CUDA device index {index} is unavailable.")
        return requested
    raise BackendError("device must be auto, cpu, mps, cuda, or cuda:<index>.")


class TransformersBackend:
    """Load only cached safetensors; download is an explicit registry operation.

    NLI uses the entailment logit (not the label index assumed across models).
    Reranking and reward backends use their single raw classification logit.
    NLI checkpoints additionally score a statement directly against the state,
    returning entailment, neutral and contradiction logits. The caller owns
    normalization and calibration.
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        revision: str,
        family: str,
        context_limit: int,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int | None = None,
        template: str = "default",
        precision: str = "float32",
        model_path: str | None = None,
    ) -> None:
        if max_length is None:
            max_length = default_max_length(context_limit)
        if batch_size < 1 or max_length < 16:
            raise BackendError("batch_size must be positive and max_length at least 16.")
        if max_length > context_limit:
            raise BackendError(f"{name} supports max_length up to {context_limit}.")
        if template not in {"default", "short"}:
            raise BackendError("template must be 'default' or 'short'.")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise BackendError("precision must be float32, float16, or bfloat16.")
        if model_path is not None:
            raise BackendError(
                "Custom PyTorch model_path is unsupported because it would misstate model identity. "
                "Use the 'onnx' backend with a verified export manifest for local artifacts."
            )
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.family = family
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.max_length = max_length
        self.template = template
        self.model_path = model_path
        self._model_file_size_bytes = MODEL_SPECS[name]["weights_bytes"]
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._entailment_index: int | None = None
        self._neutral_index: int | None = None
        self._contradiction_index: int | None = None
        self._lock = threading.RLock()
        self._truncated_candidates = 0
        self.load_time_ms: float | None = None

    @property
    def supports_statements(self) -> bool:
        """Only NLI checkpoints can judge a statement as entailed or contradicted."""
        return self.family == "nli"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "runtime": "transformers",
            "supports_statements": self.supports_statements,
            "family": self.family,
            "template": self.template,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "local_files_only": True,
            "load_time_ms": self.load_time_ms,
            "model_file_size_bytes": self._model_file_size_bytes,
            "truncated_candidates": self._truncated_candidates,
            "truncation": "state-prefix-only; full question and choice preserved",
        }

    def load(self) -> None:
        """Load once, with no remote code and no implicit download."""
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise BackendError(
                    "Local inference requires optional dependencies. "
                    "Install with: pip install -e '.[inference]'"
                ) from exc
            self.device = select_device(self.device, torch)
            if self.device == "cpu" and self.precision == "float16":
                raise BackendError("float16 on CPU is unsupported; use precision='float32'.")
            location = self.model_path or self.model_id
            options: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if not self.model_path:
                options["revision"] = self.revision
            try:
                tokenizer = AutoTokenizer.from_pretrained(location, **options)
                model_options = dict(options)
                # Disable ModernBERT's optional compilation: correctness and portability first.
                if self.name == "base":
                    model_options["reference_compile"] = False
                model = AutoModelForSequenceClassification.from_pretrained(
                    location,
                    use_safetensors=True,
                    torch_dtype=getattr(torch, self.precision),
                    attn_implementation="eager",
                    **model_options,
                )
                model.to(self.device)
                model.eval()
                if tokenizer.pad_token_id is None:
                    raise BackendError("Checkpoint tokenizer has no padding token.")
                if self.family == "nli":
                    labels = {
                        str(value).lower(): int(key) for key, value in model.config.id2label.items()
                    }
                    if "entailment" not in labels:
                        raise BackendError("NLI checkpoint has no explicit entailment label.")
                    self._entailment_index = labels["entailment"]
                    self._neutral_index = labels.get("neutral")
                    self._contradiction_index = labels.get("contradiction")
                elif model.config.num_labels != 1:
                    raise BackendError("Reward/reranker checkpoint must have one output logit.")
                if self.family == "reward" and not tokenizer.chat_template:
                    raise BackendError("Reward tokenizer requires its upstream chat template.")
            except BackendError:
                raise
            except Exception as exc:
                raise BackendError(
                    f"Cannot load cached {self.name} on {self.device}. "
                    f"Run 'opendecision pull {self.name}' first; if already cached, "
                    "check inference dependencies and try device='cpu'. "
                    f"Cause: {type(exc).__name__}: {exc}"
                ) from exc
            self._tokenizer, self._model, self._torch = tokenizer, model, torch
            self.load_time_ms = (time.perf_counter() - started) * 1000

    def ensure_loaded(self) -> None:
        """Public warm-load hook for server startup and cold-load measurements."""
        self.load()

    def _serialize(self, state: str, question: str, choice: str) -> tuple[str, str | None]:
        if self.family == "nli":
            hypothesis = (
                f"{question} {choice}"
                if self.template == "short"
                else f'Given the situation, the answer to "{question}" is "{choice}".'
            )
            return state, hypothesis
        if self.family == "reranker":
            return f"[QUESTION]\n{question}\n\n[STATE]\n{state}", choice
        prompt = (
            f"{state}\n\n{question}"
            if self.template == "short"
            else f"[STATE]\n{state}\n\n[QUESTION]\n{question}"
        )
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": choice},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return text, None

    def _token_length(self, pair: tuple[str, str | None]) -> int:
        return len(
            self._tokenizer(
                pair[0],
                text_pair=pair[1],
                add_special_tokens=self.family != "reward",
                truncation=False,
            )["input_ids"]
        )

    def _fit_candidate(self, state: str, question: str, choice: str) -> tuple[str, str | None]:
        return self._fit(
            state, lambda s: self._serialize(s, question, choice), what="Question and choice"
        )

    def _fit_statement(self, state: str, statement: str) -> tuple[str, str | None]:
        # A statement is the hypothesis itself; no template wraps it.
        return self._fit(state, lambda s: (s, statement), what="Statement")

    def _fit(
        self, state: str, build: Callable[[str], tuple[str, str | None]], *, what: str
    ) -> tuple[str, str | None]:
        """Truncate only state, reserving at least 32 state tokens where possible."""
        pair = build(state)
        if self._token_length(pair) <= self.max_length:
            return pair
        state_ids = self._tokenizer.encode(state, add_special_tokens=False)
        required = self._token_length(build(""))
        minimum_state = min(32, len(state_ids))
        if required + minimum_state > self.max_length:
            raise BackendError(
                f"{what} are too long for max_length while preserving state. "
                "Shorten them or raise max_length within the model's context limit."
            )
        budget = min(len(state_ids), self.max_length - required)
        while budget >= minimum_state:
            shortened = self._tokenizer.decode(
                state_ids[:budget], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            pair = build(shortened)
            length = self._token_length(pair)
            if length <= self.max_length:
                self._truncated_candidates += 1
                return pair
            budget -= max(1, length - self.max_length)
        raise BackendError(
            "Cannot fit input without losing the question, choice, or reserved state."
        )

    def score_choices(self, *, state: str, question: str, choices: list[str]) -> list[float]:
        # Shared implementation keeps single and batch serialization identical.
        from opendecision.schemas import DecisionRequest

        return self.score_batch([DecisionRequest(state=state, question=question, choices=choices)])[
            0
        ]

    def _infer_chunk(
        self, chunk: list[tuple[str, str | None]], columns: list[int]
    ) -> list[list[float]]:
        """Return the selected logit columns, one row per input pair."""
        encoded = self._tokenizer(
            [p[0] for p in chunk],
            text_pair=None if self.family == "reward" else [p[1] for p in chunk],
            add_special_tokens=self.family != "reward",
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
            series = [logits[:, column].detach().float().cpu().tolist() for column in columns]
        return [list(row) for row in zip(*series)]

    def _infer(
        self, pairs: list[tuple[str, str | None]], columns: list[int], preserved: str
    ) -> list[list[float]]:
        """Microbatch all pairs under the lock; fail closed on any non-finite output."""
        try:
            if self._truncated_candidates:
                warnings.warn(
                    f"State truncated for {self._truncated_candidates} candidates; "
                    f"{preserved} preserved. See result metadata.",
                    UserWarning,
                    stacklevel=3,
                )
            rows: list[list[float]] = []
            for offset in range(0, len(pairs), self.batch_size):
                rows.extend(self._infer_chunk(pairs[offset : offset + self.batch_size], columns))
            if len(rows) != len(pairs) or not all(
                math.isfinite(value) for row in rows for value in row
            ):
                raise BackendError("Backend returned missing or non-finite candidate scores.")
            return rows
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                f"{self.name} inference failed on {self.device}: {type(exc).__name__}. "
                "Try device='cpu', a smaller batch_size, or a shorter max_length."
            ) from exc

    def score_batch(self, requests: list[DecisionRequest]) -> list[list[float]]:
        if not requests:
            return []
        with self._lock:
            self.load()
            self._truncated_candidates = 0
            pairs = []
            for request in requests:
                state_text = request.state_text  # rendered once per request
                pairs.extend(
                    self._fit_candidate(state_text, request.question, candidate)
                    for candidate in request.candidate_texts
                )
            column = self._entailment_index if self.family == "nli" else 0
            scores = [row[0] for row in self._infer(pairs, [column], "question and choices")]
            grouped: list[list[float]] = []
            offset = 0
            for request in requests:
                count = len(request.choices)
                grouped.append(scores[offset : offset + count])
                offset += count
            return grouped

    def score_statements(self, requests: list[StatementRequest]) -> list[list[float]]:
        """Return [entailment, neutral, contradiction] logits per statement."""
        if not requests:
            return []
        with self._lock:
            self.load()
            if not self.supports_statements or None in (
                self._neutral_index,
                self._contradiction_index,
            ):
                raise BackendError(
                    f"{self.name} cannot score statements: this requires an NLI checkpoint "
                    "with entailment, neutral, and contradiction labels."
                )
            self._truncated_candidates = 0
            pairs = [self._fit_statement(r.state_text, r.statement) for r in requests]
            columns = [self._entailment_index, self._neutral_index, self._contradiction_index]
            return self._infer(pairs, columns, "statements")

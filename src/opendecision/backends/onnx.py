"""CPU ONNX inference for locally exported, content-addressed NLI artifacts."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opendecision.backends.catalog import MODEL_SPECS, backend_options
from opendecision.backends.transformers import TransformersBackend
from opendecision.errors import BackendError

if TYPE_CHECKING:
    from opendecision.schemas import DecisionRequest


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identity(manifest: dict[str, Any]) -> str:
    """Content identity binds the graph, tokenizer, source pin and quantization."""
    content = {key: manifest[key] for key in ("format", "source", "precision", "files")}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_manifest(directory: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "opendecision-manifest.json").read_text())
        if manifest["format"] != "opendecision-onnx-v1":
            raise ValueError("unsupported manifest format")
        source = manifest["source"]
        spec = MODEL_SPECS[source["name"]]
        if source["name"] != "tiny":
            raise ValueError("alpha ONNX export supports only the verified tiny architecture")
        if any(source[key] != spec[key] for key in ("model_id", "revision")):
            raise ValueError("source identity does not match the audited checkpoint")
        if manifest["precision"] not in {"float32", "int8-dynamic"}:
            raise ValueError("unsupported precision")
        files = manifest["files"]
        if (
            not {"model.onnx", "config.json", "tokenizer_config.json", "tokenizer.json"}
            <= files.keys()
        ):
            raise ValueError("manifest is missing required model or tokenizer files")
        for filename, digest in files.items():
            if Path(filename).name != filename or filename in {".", ".."}:
                raise ValueError("manifest paths must be plain filenames")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid file digest")
        if manifest["artifact_sha256"] != manifest_identity(manifest):
            raise ValueError("manifest identity mismatch")
        return manifest
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BackendError(
            "Invalid ONNX artifact. Export it with scripts/export_onnx.py; "
            f"manifest validation failed: {exc}"
        ) from exc


class OnnxBackend(TransformersBackend):
    """Use ONNX Runtime CPU with the identical serialization as the source model."""

    def __init__(self, *, model_path: str, device: str = "cpu", **kwargs: Any) -> None:
        if device not in {"auto", "cpu"}:
            raise BackendError("The alpha ONNX backend supports device='cpu' only.")
        self._directory = Path(model_path).expanduser().resolve()
        self._manifest = read_manifest(self._directory)
        source = self._manifest["source"]
        requested_precision = kwargs.pop("precision", None)
        if requested_precision is not None and requested_precision != self._manifest["precision"]:
            raise BackendError("ONNX precision is fixed by its export manifest.")
        super().__init__(**backend_options(source["name"]), device="cpu", **kwargs)
        self.name = "onnx"
        self.model_id = source["model_id"] + "/onnx"
        self.revision = "sha256:" + self._manifest["artifact_sha256"]
        self.precision = self._manifest["precision"]
        self._model_file_size_bytes = None
        self._input_names: set[str] = set()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "runtime": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "source_model": self._manifest["source"],
            "artifact_sha256": self._manifest["artifact_sha256"],
        }

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            # Hash all tokenizer/graph files as well as binding their names in the identity.
            # Run before library imports so tampering fails quickly and predictably.
            for filename, expected in self._manifest["files"].items():
                path = self._directory / filename
                if not path.is_file() or path.is_symlink() or hash_file(path) != expected:
                    raise BackendError(f"ONNX artifact integrity check failed for {filename}.")
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise BackendError(
                    "Install ONNX support: pip install -e '.[inference,onnx]'"
                ) from exc
            try:
                config = json.loads((self._directory / "config.json").read_text())
                labels = {value.lower(): int(key) for key, value in config["id2label"].items()}
                self._entailment_index = labels["entailment"]
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self._directory), local_files_only=True, trust_remote_code=False
                )
                session = ort.InferenceSession(
                    str(self._directory / "model.onnx"), providers=["CPUExecutionProvider"]
                )
                output_names = [output.name for output in session.get_outputs()]
                if "logits" not in output_names:
                    raise ValueError("ONNX graph must expose logits output")
                self._input_names = {field.name for field in session.get_inputs()}
                self._model_file_size_bytes = sum(
                    (self._directory / filename).stat().st_size
                    for filename in self._manifest["files"]
                    if filename.endswith((".onnx", ".onnx_data", ".data"))
                )
            except Exception as exc:
                raise BackendError(
                    f"Cannot load ONNX artifact: {type(exc).__name__}: {exc}"
                ) from exc
            self._tokenizer, self._model = tokenizer, session
            self.load_time_ms = (time.perf_counter() - started) * 1000

    def _infer_chunk(self, chunk: list[tuple[str, str | None]]) -> list[float]:
        encoded = self._tokenizer(
            [pair[0] for pair in chunk],
            text_pair=[pair[1] for pair in chunk],
            padding=True,
            truncation=False,
            return_tensors="np",
        )
        inputs = {name: value for name, value in encoded.items() if name in self._input_names}
        logits = self._model.run(["logits"], inputs)[0]
        return logits[:, self._entailment_index].astype(float).tolist()

    def score_batch(self, requests: list[DecisionRequest]) -> list[list[float]]:
        """Score flattened candidate pairs through ONNX Runtime, preserving boundaries."""
        if not requests:
            return []
        with self._lock:
            self.load()
            self._truncated_candidates = 0
            pairs = [
                self._fit_candidate(request.state, request.question, choice)
                for request in requests
                for choice in request.choices
            ]
            if self._truncated_candidates:
                warnings.warn(
                    f"State truncated for {self._truncated_candidates} candidates; "
                    "question and choices preserved. See result metadata.",
                    UserWarning,
                    stacklevel=2,
                )
            scores: list[float] = []
            for offset in range(0, len(pairs), self.batch_size):
                scores.extend(self._infer_chunk(pairs[offset : offset + self.batch_size]))
            if len(scores) != len(pairs):
                raise BackendError("ONNX returned the wrong number of candidate scores.")
            grouped: list[list[float]] = []
            offset = 0
            for request in requests:
                count = len(request.choices)
                grouped.append(scores[offset : offset + count])
                offset += count
            return grouped

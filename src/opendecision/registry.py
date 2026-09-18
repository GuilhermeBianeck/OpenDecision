"""Model catalog and explicit, pinned downloads; normal inference stays offline."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from opendecision.backends.base import DecisionBackend
from opendecision.backends.catalog import MODEL_SPECS, VERIFIED_DATE, backend_options
from opendecision.errors import BackendError

# Do not fetch pickle weights, Python modules, unrelated ONNX variants, or assets.
DOWNLOAD_PATTERNS = [
    ".gitattributes",
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "README.md",
    "LICENSE*",
    "NOTICE*",
]


def default_model(device: str = "auto") -> str:
    """The model ``auto`` resolves to on this machine.

    Apple silicon gets the MLX decoder, which measured 0.799 objective accuracy
    against 0.675 for the portable baseline on the committed test split. Every
    other platform keeps that baseline, which needs no MLX runtime.
    """
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return "qwen35" if apple and device in {"auto", "mps"} else "base"


def _resolve_name(name: str, device: str = "auto") -> str:
    aliases = {
        "auto": "base",
        "bge": "multilingual",
        "modernbert": "base",
        "deberta": "tiny",
        "qwen": "decoder",
        "qwen3.5": "qwen35",
        "lfm": "lfm25",
    }
    if name == "auto":
        return default_model(device)
    resolved = aliases.get(name, name)
    if resolved not in {*MODEL_SPECS, "demo", "onnx"}:
        raise BackendError(
            f"Unknown model {name!r}. Choose tiny, base, smart, multilingual, decoder, qwen35, lfm25, "
            "demo, or onnx."
        )
    return resolved


def list_models() -> list[dict[str, Any]]:
    """Return a copy of model facts without importing torch or accessing the network."""
    models = [{**spec, "license_verified_at": VERIFIED_DATE} for spec in MODEL_SPECS.values()]
    models.append(
        {
            "name": "demo",
            "model_id": "opendecision/token-overlap-demo",
            "revision": "demo-v1",
            "family": "infrastructure-fixture",
            "license": "MIT",
            "parameters": 0,
            "weights_bytes": 0,
            "description": "Deterministic token overlap; no learned decision quality.",
        }
    )
    models.append(
        {
            "name": "onnx",
            "model_id": "local-export",
            "revision": "content-addressed at export",
            "family": "nli",
            "license": "inherits source checkpoint",
            "parameters": None,
            "weights_bytes": None,
            "description": "CPU ONNX export; requires model_path with an export manifest.",
        }
    )
    return models


def create_backend(name: str = "base", device: str = "auto", **kwargs: Any) -> DecisionBackend:
    """Construct a lazy backend; auto prefers Qwen3.5 on Apple silicon."""
    if name == "auto":
        resolved = (
            "qwen35" if platform.system() == "Darwin" and device in {"auto", "mps"} else "base"
        )
    else:
        resolved = _resolve_name(name)
    if resolved == "onnx":
        from opendecision.backends.onnx import OnnxBackend

        if not kwargs.get("model_path"):
            raise BackendError("ONNX requires model_path pointing to a verified export directory.")
        return OnnxBackend(device=device, **kwargs)
    if resolved == "demo":
        from opendecision.backends.demo import DemoBackend

        return DemoBackend()
    if resolved == "decoder":
        from opendecision.backends.decoder import DecoderBackend

        return DecoderBackend(**backend_options("decoder"), device=device, **kwargs)
    if resolved in {"qwen35", "lfm25", "qwen35_4b", "lfm25_26b"}:
        from opendecision.backends.mlx_decoder import MLXDecoderBackend

        return MLXDecoderBackend(**backend_options(resolved), device=device, **kwargs)
    from opendecision.backends.bge_reranker import BGERerankerBackend
    from opendecision.backends.deberta import DebertaBackend
    from opendecision.backends.modernbert import ModernBertBackend
    from opendecision.backends.skywork_reward import SkyworkRewardBackend

    constructors = {
        "tiny": DebertaBackend,
        "base": ModernBertBackend,
        "smart": SkyworkRewardBackend,
        "multilingual": BGERerankerBackend,
    }
    return constructors[resolved](device=device, **kwargs)


def pull_model(name: str, device: str = "cpu") -> dict[str, Any]:
    """Explicitly download an audited revision, then load and smoke-test it offline.

    Files use the standard Hugging Face cache (including HF_HOME/HF_HUB_CACHE).
    No user input is ever sent to the Hub; the smoke input is a public fixture.
    """
    resolved = _resolve_name(name, device)
    if resolved == "onnx":
        raise BackendError("ONNX artifacts are local exports. Use scripts/export_onnx.py first.")
    if resolved == "demo":
        return {"name": "demo", "disk_bytes": 0, "verified": True, "infrastructure_only": True}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BackendError(
            "Install download/inference support: pip install -e '.[inference]'"
        ) from exc
    spec = MODEL_SPECS[resolved]
    try:
        path = Path(
            snapshot_download(
                repo_id=spec["model_id"],
                revision=spec["revision"],
                allow_patterns=DOWNLOAD_PATTERNS,
            )
        )
    except Exception as exc:
        raise BackendError(
            f"Could not download pinned {resolved}: {type(exc).__name__}. "
            "Check network connectivity and available disk space, then retry."
        ) from exc
    if not any(path.glob("*.safetensors")):
        raise BackendError("Downloaded snapshot contains no safetensors weights.")
    backend = create_backend(resolved, device=device)
    scores = backend.score_choices(
        state="The customer reports a duplicate card charge.",
        question="Which team should handle this?",
        choices=["billing support", "technical support"],
    )
    # Resolve symlinks before counting: HF snapshots refer to shared blob storage.
    files = {file.resolve() for file in path.rglob("*") if file.is_file()}
    return {
        **spec,
        "license_verified_at": VERIFIED_DATE,
        "path": str(path),
        "disk_bytes": sum(file.stat().st_size for file in files),
        "smoke_scores": scores,
        "verified": True,
        "device": backend.device,
        "precision": backend.precision,
    }

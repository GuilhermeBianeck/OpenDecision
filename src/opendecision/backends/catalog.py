"""Audited upstream model identities. Never follow a moving branch at inference."""

from typing import Any

VERIFIED_DATE = "2026-09-18"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tiny": {
        "name": "tiny",
        "model_id": "cross-encoder/nli-deberta-v3-xsmall",
        "revision": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
        "family": "nli",
        "license": "Apache-2.0",
        "parameters": 70_831_107,
        "weights_bytes": 283_353_172,
        "context_limit": 512,
        "description": "DeBERTa-v3-xsmall NLI baseline; English.",
    },
    "base": {
        "name": "base",
        "model_id": "tasksource/ModernBERT-base-nli",
        "revision": "de4ab7e77845098b7fab7f6ab9d370ddff27b19c",
        "family": "nli",
        "license": "Apache-2.0",
        "parameters": 149_607_171,
        "weights_bytes": 598_442_860,
        "context_limit": 2048,
        "description": "ModernBERT NLI baseline; English; default for auto.",
    },
    "smart": {
        "name": "smart",
        "model_id": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        "revision": "8c14a4e9e6321deaf572544339b16b8d6bbe8886",
        "family": "reward",
        "license": "Apache-2.0",
        "parameters": 596_050_944,
        "weights_bytes": 1_192_137_232,
        # Upstream recommends remaining within its 16,384-token training length.
        "context_limit": 16384,
        "description": "Skywork Qwen3 reward scorer; no text generation.",
    },
    "decoder": {
        "name": "decoder",
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "family": "decoder",
        "license": "Apache-2.0",
        "parameters": 596_049_920,
        "weights_bytes": 1_503_300_328,
        # Upstream context is 32,768 tokens; the default limit caps at 2,048.
        "context_limit": 32768,
        "description": "Qwen3 0.6B decoder; encodes each state once and reads option letters.",
    },
    "multilingual": {
        "name": "multilingual",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "family": "reranker",
        "license": "Apache-2.0",
        "parameters": 567_755_777,
        "weights_bytes": 2_271_071_852,
        "context_limit": 8192,
        "description": "Multilingual BGE relevance scorer; experimental decision adapter.",
    },
}


def backend_options(name: str) -> dict[str, Any]:
    """Select model constructor fields, excluding display-only metadata."""
    spec = MODEL_SPECS[name]
    return {key: spec[key] for key in ("name", "model_id", "revision", "family", "context_limit")}

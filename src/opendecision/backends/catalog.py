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
        "description": "ModernBERT NLI baseline; English; portable CPU fallback for auto.",
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
    "qwen35": {
        "name": "qwen35",
        "model_id": "mlx-community/Qwen3.5-2B-4bit",
        "revision": "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
        "family": "decoder-mlx",
        "license": "Apache-2.0",
        "parameters": 2_000_000_000,
        "weights_bytes": 1_722_271_785,
        "context_limit": 8192,
        "description": "Qwen3.5 2B, MLX affine Q4; general local decision candidate.",
    },
    "lfm25": {
        "name": "lfm25",
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "revision": "7ccafdb04c36936f4f1c4685198c6c9a40275932",
        "family": "decoder-mlx",
        "license": "LFM-1.0",
        "parameters": 1_200_000_000,
        "weights_bytes": 658_540_250,
        "context_limit": 8192,
        "description": "LFM2.5 1.2B instruct, MLX affine Q4; fast local decision candidate.",
    },
    "qwen35_4b": {
        "name": "qwen35_4b",
        "model_id": "mlx-community/Qwen3.5-4B-4bit",
        "revision": "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
        "family": "decoder-mlx",
        "license": "Apache-2.0",
        "parameters": 4_000_000_000,
        "weights_bytes": 3_034_300_695,
        "context_limit": 8192,
        "description": "Qwen3.5 4B, MLX affine Q4; quality-focused local decision candidate.",
    },
    "lfm25_26b": {
        "name": "lfm25_26b",
        "model_id": "LiquidAI/LFM2.5-2.6B-MLX-4bit",
        "revision": "04efa23776ce61ec34ec95ec34c859854c89542b",
        "family": "decoder-mlx",
        "license": "LFM-1.0",
        "parameters": 2_600_000_000,
        "weights_bytes": 1_583_152_892,
        "context_limit": 8192,
        "description": "LFM2.5 2.6B, MLX affine Q4; larger quality candidate.",
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

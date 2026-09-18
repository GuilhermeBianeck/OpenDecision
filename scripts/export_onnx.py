#!/usr/bin/env python3
"""Export the pinned tiny NLI baseline to a content-addressed ONNX artifact.

This is an explicit, optional operation. It requires ``.[inference,onnx]`` and
an already cached model; it never downloads weights or replaces a registered
model. The resulting directory can be passed to ``DecisionModel('onnx',
model_path='...')`` from Python. The CLI currently requires an application-level
path option for custom ONNX artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: str, output: Path, precision: str = "float32") -> dict[str, Any]:
    if source != "tiny":
        raise ValueError("The alpha exporter supports only the pinned tiny NLI model")
    if precision not in {"float32", "int8-dynamic"}:
        raise ValueError("precision must be float32 or int8-dynamic")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install optional export dependencies with pip install -e '.[inference,onnx]'"
        ) from exc

    from huggingface_hub import snapshot_download

    from opendecision.backends.catalog import MODEL_SPECS

    spec = MODEL_SPECS[source]
    snapshot = Path(
        snapshot_download(
            spec["model_id"],
            revision=spec["revision"],
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors",
                "README.md",
            ],
            local_files_only=True,
        )
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True
    ).eval()
    sample = tokenizer(
        "The account has a duplicate payment.", "This is a billing issue.", return_tensors="pt"
    )
    allowed = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in sample]

    class Wrapper(torch.nn.Module):
        def forward(self, *values):
            inputs = dict(zip(allowed, values, strict=True))
            return model(**inputs).logits

    wrapper = Wrapper()
    onnx_path = output / "model.onnx"
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in allowed}
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        tuple(sample[name] for name in allowed),
        onnx_path,
        input_names=allowed,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    if precision == "int8-dynamic":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to create an INT8 artifact") from exc
        quantized = output / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(quantized), weight_type=QuantType.QInt8)
        onnx_path.unlink()
        quantized.rename(onnx_path)

    copied = {"model.onnx": onnx_path}
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        source_path = snapshot / filename
        if source_path.is_file():
            shutil.copy2(source_path, output / filename)
            copied[filename] = output / filename
    files = {name: _sha256(path) for name, path in copied.items()}
    manifest: dict[str, Any] = {
        "format": "opendecision-onnx-v1",
        "source": {"name": source, "model_id": spec["model_id"], "revision": spec["revision"]},
        "precision": precision,
        "files": files,
    }
    manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            {key: manifest[key] for key in ("format", "source", "precision", "files")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path = output / "opendecision-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tiny")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", default="float32", choices=("float32", "int8-dynamic"))
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.precision), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys

import pytest

from opendecision.backends.base import DecisionBackend
from opendecision.backends.catalog import MODEL_SPECS
from opendecision.errors import BackendError
from opendecision.registry import DOWNLOAD_PATTERNS, create_backend, list_models, pull_model


def test_catalog_uses_immutable_unique_revisions_and_has_no_shared_mutation():
    models = list_models()
    assert {model["name"] for model in models} == {
        "tiny",
        "base",
        "smart",
        "multilingual",
        "demo",
        "onnx",
    }
    for model in models:
        if model["name"] not in {"demo", "onnx"}:
            assert len(model["revision"]) == 40
            int(model["revision"], 16)
            assert model["license"] == "Apache-2.0"
    models[0]["revision"] = "changed"
    assert list_models()[0]["revision"] != "changed"


def test_creating_backend_does_not_import_heavy_dependencies():
    script = (
        "import sys; from opendecision.registry import create_backend; "
        "b=create_backend('base'); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
        "assert b._model is None"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_auto_alias_has_stable_identity():
    backend = create_backend("auto")
    assert backend.name == "base"
    assert backend.model_id == MODEL_SPECS["base"]["model_id"]
    assert isinstance(backend, DecisionBackend)


def test_unknown_backend_is_actionable():
    with pytest.raises(BackendError, match="Choose tiny, base, smart"):
        create_backend("unknown")


def test_demo_is_explicit_infrastructure_only_and_order_equivariant():
    demo = create_backend("demo")
    choices = ["billing", "technical"]
    forward = demo.score_choices(state="billing", question="Which team?", choices=choices)
    reverse = demo.score_choices(state="billing", question="Which team?", choices=choices[::-1])
    assert forward == reverse[::-1]
    assert demo.metadata["infrastructure_only"]
    assert pull_model("demo")["disk_bytes"] == 0


def test_download_allowlist_excludes_pickle_and_code():
    assert "model.safetensors" in DOWNLOAD_PATTERNS
    assert "chat_template.jinja" in DOWNLOAD_PATTERNS
    assert not any(pattern.endswith((".bin", ".py", ".onnx")) for pattern in DOWNLOAD_PATTERNS)

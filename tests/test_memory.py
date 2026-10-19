"""Refusing to load a model the machine cannot hold."""

from __future__ import annotations

import pytest

from opendecision import memory
from opendecision.backends.catalog import MODEL_SPECS
from opendecision.errors import BackendError
from opendecision.memory import RESERVE_BYTES, check_memory, required_bytes
from opendecision.registry import create_backend


def test_measured_peaks_take_precedence_over_the_weight_estimate():
    spec = MODEL_SPECS["qwen35_4b"]
    assert required_bytes(spec) == spec["memory_bytes"] + memory.PROCESS_OVERHEAD_BYTES
    unmeasured = {"name": "x", "weights_bytes": 1_000}
    assert required_bytes(unmeasured) == 2_000 + memory.PROCESS_OVERHEAD_BYTES


def test_every_mlx_backend_that_was_measured_fits_a_16_gb_machine():
    for spec in MODEL_SPECS.values():
        if spec.get("memory_bytes") is not None:
            assert required_bytes(spec) + RESERVE_BYTES < 8e9


def test_enough_memory_loads_and_too_little_refuses_with_alternatives():
    spec = MODEL_SPECS["qwen35_4b"]
    check_memory(spec, available=required_bytes(spec) + RESERVE_BYTES)
    with pytest.raises(BackendError) as raised:
        check_memory(spec, available=required_bytes(MODEL_SPECS["qwen35"]) + RESERVE_BYTES)
    message = str(raised.value)
    assert "Not enough free memory to load qwen35_4b" in message
    assert "qwen35" in message.split("fit now:")[1]
    assert "OPENDECISION_SKIP_MEMORY_CHECK=1" in message


def test_nothing_fits_suggests_closing_applications():
    with pytest.raises(BackendError, match="Close other applications"):
        check_memory(MODEL_SPECS["tiny"], available=1)


def test_unknown_availability_and_the_override_do_not_block(monkeypatch):
    monkeypatch.setattr(memory, "available_bytes", lambda: None)
    check_memory(MODEL_SPECS["qwen35_4b"])
    monkeypatch.setattr(memory, "available_bytes", lambda: 1)
    monkeypatch.setenv("OPENDECISION_SKIP_MEMORY_CHECK", "1")
    check_memory(MODEL_SPECS["qwen35_4b"])


def test_this_machine_reports_available_memory():
    value = memory.available_bytes()
    if value is None:
        pytest.skip("No available-memory source on this platform")
    assert value > 0


def test_a_backend_refuses_before_loading_weights(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(memory, "available_bytes", lambda: 1)
    monkeypatch.delenv("OPENDECISION_SKIP_MEMORY_CHECK", raising=False)
    backend = create_backend("tiny", device="cpu")
    with pytest.raises(BackendError, match="Not enough free memory to load tiny"):
        backend.load()

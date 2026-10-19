"""Long prompts are encoded in bounded chunks, ending with the scored tokens."""

from __future__ import annotations

from types import SimpleNamespace

from opendecision.backends import mlx_decoder
from opendecision.registry import create_backend


class Recorder:
    def __init__(self):
        self.calls = []

    def model(self, batch, cache=None):
        self.calls.append(len(batch[0]))
        return self

    def __getitem__(self, _):
        return self


def backend_with(recorder):
    backend = create_backend("qwen35")
    backend._lm = SimpleNamespace(model=recorder.model, lm_head=lambda hidden: hidden)
    backend._mx = SimpleNamespace(array=lambda rows: rows, eval=lambda *a: None)
    return backend


def test_long_prompts_are_encoded_in_chunks(monkeypatch):
    monkeypatch.setattr(mlx_decoder, "PREFILL_CHUNK", 4)
    recorder = Recorder()
    backend_with(recorder)._last_logits(list(range(10)), cache=[])
    # Two full chunks of context, then the last two tokens produce the logits.
    assert recorder.calls == [4, 4, 2]


def test_short_prompts_take_one_pass(monkeypatch):
    monkeypatch.setattr(mlx_decoder, "PREFILL_CHUNK", 4)
    recorder = Recorder()
    backend_with(recorder)._last_logits(list(range(4)), cache=[])
    assert recorder.calls == [4]


def test_a_shared_prefix_is_chunked_too(monkeypatch):
    monkeypatch.setattr(mlx_decoder, "PREFILL_CHUNK", 4)
    recorder = Recorder()
    backend_with(recorder)._prefill(list(range(9)), cache=[])
    assert recorder.calls == [4, 4, 1]

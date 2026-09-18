"""Dependency-free adapter contract tests; real weights are opt-in below."""

from __future__ import annotations

import contextlib
import os
import sys
from types import SimpleNamespace

import pytest

from opendecision.backends.deberta import DebertaBackend
from opendecision.backends.modernbert import ModernBertBackend
from opendecision.backends.skywork_reward import SkyworkRewardBackend
from opendecision.backends.transformers import select_device
from opendecision.errors import BackendError
from opendecision.registry import create_backend
from opendecision.schemas import DecisionRequest, StatementRequest


class FakeEncoded(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    """Word tokenizer sufficient to check batching and preserved input content."""

    def __init__(self):
        self.batches = []
        self.chats = []

    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, ids, **kwargs):
        return " ".join(ids)

    def apply_chat_template(self, messages, **kwargs):
        self.chats.append(messages)
        return " ".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, text_pair=None, **kwargs):
        if isinstance(text, str):
            return {"input_ids": text.split() + (text_pair or "").split() + ["SEP"]}
        self.batches.append((text, text_pair, kwargs))
        return FakeEncoded(input_ids=list(range(len(text))))


class FakeValues:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeLogits:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        _, column = key
        return FakeValues([row[column] for row in self.rows])


class FakeModel:
    def __init__(self):
        self.offset = 0
        self.calls = []

    def __call__(self, input_ids):
        self.calls.append(len(input_ids))
        rows = [[-999, float(i), 999] for i in range(self.offset, self.offset + len(input_ids))]
        self.offset += len(input_ids)
        return SimpleNamespace(logits=FakeLogits(rows))


def initialized_backend(**kwargs):
    backend = DebertaBackend(**kwargs)
    backend._tokenizer = FakeTokenizer()
    backend._model = FakeModel()
    backend._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    backend._entailment_index = 1
    backend._neutral_index = 0
    backend._contradiction_index = 2
    return backend


def test_flattened_batches_restore_request_boundaries_and_entailment_column():
    backend = initialized_backend(batch_size=3)
    requests = [
        DecisionRequest(state="s1", question="q1", choices=["a", "b"]),
        DecisionRequest(state="s2", question="q2", choices=["c", "d", "e"]),
    ]
    assert backend.score_batch(requests) == [[0.0, 1.0], [2.0, 3.0, 4.0]]
    assert backend._model.calls == [3, 2]
    assert all(batch[2]["truncation"] is False for batch in backend._tokenizer.batches)


def test_structured_state_and_option_text_are_serialized_for_the_model():
    backend = initialized_backend()
    backend.score_batch(
        [
            DecisionRequest(
                state={"ticket": "card charged twice", "tier": "gold"},
                question="Which team?",
                choices=["technical", {"label": "billing", "description": "payments and refunds"}],
            )
        ]
    )
    premises, hypotheses, _ = backend._tokenizer.batches[0]
    assert premises == ["ticket: card charged twice\ntier: gold"] * 2
    assert "billing: payments and refunds" in hypotheses[1]
    assert "technical" in hypotheses[0]


def test_truncation_retains_whole_question_and_choice_and_warns():
    backend = initialized_backend(max_length=64)
    with pytest.warns(UserWarning, match="State truncated for 2"):
        backend.score_choices(
            state=" ".join(f"state{i}" for i in range(100)),
            question="the complete decision question",
            choices=["first candidate stays complete", "second candidate also complete"],
        )
    premises, hypotheses, _ = backend._tokenizer.batches[0]
    assert len(premises[0].split()) >= 32
    assert "state99" not in premises[0]
    assert "the complete decision question" in hypotheses[0]
    assert "first candidate stays complete" in hypotheses[0]
    assert "second candidate also complete" in hypotheses[1]
    assert backend.metadata["truncated_candidates"] == 2


def test_statements_return_entailment_neutral_contradiction_in_order():
    backend = initialized_backend(batch_size=2)
    assert backend.supports_statements
    rows = backend.score_statements(
        [
            StatementRequest(state="s1", statement="the first claim"),
            StatementRequest(state="s2", statement="another claim"),
            StatementRequest(state="s3", statement="third"),
        ]
    )
    # FakeModel emits [-999, i, 999]; columns are (entailment=1, neutral=0, contradiction=2).
    assert rows == [[0.0, -999.0, 999.0], [1.0, -999.0, 999.0], [2.0, -999.0, 999.0]]
    assert backend._model.calls == [2, 1]
    premises, hypotheses, _ = backend._tokenizer.batches[0]
    assert hypotheses == ["the first claim", "another claim"]
    assert premises == ["s1", "s2"]


def test_statement_state_truncation_preserves_the_statement():
    backend = initialized_backend(max_length=40)
    with pytest.warns(UserWarning, match="statements preserved"):
        rows = backend.score_statements(
            [StatementRequest(state=" ".join(f"w{i}" for i in range(200)), statement="keep me")]
        )
    assert len(rows) == 1
    premises, hypotheses, _ = backend._tokenizer.batches[0]
    assert hypotheses == ["keep me"]
    assert "w199" not in premises[0]
    assert backend.metadata["truncated_candidates"] == 1


def test_non_nli_backends_decline_statements():
    reward = SkyworkRewardBackend()
    assert not reward.supports_statements
    nli_without_labels = initialized_backend()
    nli_without_labels._contradiction_index = None
    with pytest.raises(BackendError, match="cannot score statements"):
        nli_without_labels.score_statements([StatementRequest(state="s", statement="c")])
    assert SkyworkRewardBackend().score_statements([]) == []


def test_candidate_too_long_is_rejected_instead_of_silently_truncated():
    backend = initialized_backend(max_length=64)
    with pytest.raises(BackendError, match="Question and choice are too long"):
        backend.score_choices(
            state=" ".join(["state"] * 100),
            question="Which?",
            choices=[" ".join(["choice"] * 100), "other"],
        )
    assert not backend._model.calls


def test_short_nli_template_and_reward_upstream_chat_roles():
    backend = ModernBertBackend(template="short")
    assert backend._serialize("state", "Which?", "billing") == ("state", "Which? billing")
    reward = SkyworkRewardBackend()
    reward._tokenizer = FakeTokenizer()
    text, second = reward._serialize("state", "Which?", "billing")
    assert second is None
    assert "billing" in text
    assert [m["role"] for m in reward._tokenizer.chats[-1]] == ["user", "assistant"]
    assert reward._tokenizer.chats[-1][-1]["content"] == "billing"


def test_empty_batch_does_not_load_model():
    assert DebertaBackend().score_batch([]) == []


def test_device_selection_is_conservative_on_apple_and_explicit_errors():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    assert select_device("auto", torch) == "cpu"
    assert select_device("mps", torch) == "mps"
    with pytest.raises(BackendError, match="CUDA is unavailable"):
        select_device("cuda", torch)
    with pytest.raises(BackendError, match="device must be"):
        select_device("invented", torch)


@pytest.mark.parametrize(
    "backend_type,entailment_index", [(DebertaBackend, 1), (ModernBertBackend, 0)]
)
def test_load_uses_only_pinned_offline_safetensors_and_configured_labels(
    backend_type, entailment_index, monkeypatch
):
    tokenizer_calls, model_calls = [], []
    tokenizer = SimpleNamespace(pad_token_id=0)
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={entailment_index: "entailment"}),
        to=lambda device: None,
        eval=lambda: None,
    )

    def load_tokenizer(location, **kwargs):
        tokenizer_calls.append((location, kwargs))
        return tokenizer

    def load_model(location, **kwargs):
        model_calls.append((location, kwargs))
        return model

    fake_torch = SimpleNamespace(float32="float32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    backend = backend_type(device="cpu")
    assert backend.load_time_ms is None
    backend.ensure_loaded()
    assert backend.load_time_ms >= 0
    assert backend._entailment_index == entailment_index
    for _, kwargs in [*tokenizer_calls, *model_calls]:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == backend.revision
    assert model_calls[0][1]["use_safetensors"] is True
    backend.ensure_loaded()
    assert len(model_calls) == 1


@pytest.mark.parametrize("kwargs", [{"max_length": 513}, {"batch_size": 0}, {"template": "bad"}])
def test_invalid_backend_configuration(kwargs):
    with pytest.raises(BackendError):
        DebertaBackend(**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["tiny", "base", "smart", "multilingual"])
def test_pinned_cached_model_real_inference_offline(name, monkeypatch):
    """Set OPENDECISION_TEST_MODELS=tiny,base after explicit model pulls."""
    enabled = os.environ.get("OPENDECISION_TEST_MODELS", "").split(",")
    if name not in enabled:
        pytest.skip("Real weights not requested via OPENDECISION_TEST_MODELS")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    backend = create_backend(name, device="cpu", batch_size=2)
    scores = backend.score_choices(
        state="The customer reports a duplicate credit card charge.",
        question="Which team should handle this?",
        choices=["billing", "technical"],
    )
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    # Infrastructure smoke: ranking and accuracy are measured in benchmarks.
    assert backend.device == "cpu"
    assert backend.precision == "float32"

"""Prefix-cached decoder logic with fakes; real weights are opt-in in test_backends."""

from __future__ import annotations

import pytest

from opendecision.backends.catalog import backend_options
from opendecision.backends.decoder import LETTERS, DecoderBackend
from opendecision.errors import BackendError
from opendecision.registry import create_backend
from opendecision.schemas import DecisionRequest, StatementRequest


class FakeTokenizer:
    """Whitespace tokenizer; each distinct word is one id, letters are stable ids."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {letter: index for index, letter in enumerate(LETTERS)}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.replace("\n", " \n ").split(" "):
            if word == "":
                continue
            ids.append(self.vocabulary.setdefault(word, len(self.vocabulary)))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        reverse = {index: word for word, index in self.vocabulary.items()}
        return " ".join(reverse[i] for i in ids)


class FakeCache:
    def __init__(self, length):
        self.length = length
        self.crops = []

    def crop(self, length):
        self.crops.append(length)
        self.length = length


def fake_backend(**kwargs):
    backend = DecoderBackend(**backend_options("decoder"), device="cpu", **kwargs)
    backend._tokenizer = FakeTokenizer()
    backend._model = object()
    backend._letter_ids = [backend._tokenizer.vocabulary[letter] for letter in LETTERS]
    backend.prefixes = []
    backend.suffixes = []

    def prefix_cache(prefix_ids):
        backend.prefixes.append(list(prefix_ids))
        return FakeCache(len(prefix_ids))

    def letter_logprobs(cache, prefix_length, suffix_ids):
        assert cache.length == prefix_length
        backend.suffixes.append(backend._tokenizer.decode(suffix_ids))
        cache.length += len(suffix_ids)
        cache.crop(prefix_length)
        # The option whose text contains "best" wins; the position is irrelevant.
        text = backend.suffixes[-1]
        scores = [-5.0] * len(LETTERS)
        for line in text.split(" \n "):
            for index, letter in enumerate(LETTERS):
                if line.startswith(f"{letter}.") and "best" in line:
                    scores[index] = -0.1
        return scores

    backend._prefix_cache = prefix_cache
    backend._letter_logprobs = letter_logprobs
    return backend


def test_shared_state_is_encoded_once_per_group_and_cache_is_rewound():
    backend = fake_backend()
    requests = [
        DecisionRequest(state="shared context", question="First?", choices=["a", "best"]),
        DecisionRequest(state="other context", question="Second?", choices=["best", "b", "c"]),
        DecisionRequest(state="shared context", question="Third?", choices=["x", "y", "best"]),
    ]
    rows = backend.score_batch(requests)
    assert len(backend.prefixes) == 2, "one prefix encoding per distinct state"
    assert len(backend.suffixes) == 3, "one question forward per request"
    assert [row.index(max(row)) for row in rows] == [1, 0, 2]
    assert all(len(row) == len(request.choices) for row, request in zip(rows, requests))
    assert "Context: \n shared context" in backend._tokenizer.decode(backend.prefixes[0])
    assert "A. a" in backend.suffixes[0] and "B. best" in backend.suffixes[0]
    assert backend.metadata["shared_state_encoding"] is True


def test_permutations_average_over_rotated_option_orders():
    backend = fake_backend(permutations=3)
    [row] = backend.score_batch(
        [DecisionRequest(state="s", question="Q?", choices=["a", "best", "c"])]
    )
    assert len(backend.suffixes) == 3
    positions = [
        suffix.split(" \n ").index(next(line for line in suffix.split(" \n ") if "best" in line))
        for suffix in backend.suffixes
    ]
    assert len(set(positions)) == 3, "the winning option visited every position"
    assert row.index(max(row)) == 1
    assert row[1] == pytest.approx(-0.1) and row[0] == pytest.approx(-5.0)


def test_statements_map_to_supported_not_addressed_contradicted():
    backend = fake_backend()
    rows = backend.score_statements(
        [
            StatementRequest(state="s", statement="best claim"),
            StatementRequest(state="s", statement="other"),
        ]
    )
    assert len(backend.prefixes) == 1
    assert all(len(row) == 3 for row in rows)
    assert "A. true" in backend.suffixes[0] and "B. unknown" in backend.suffixes[0]
    assert "C. false" in backend.suffixes[0]
    assert "best claim" in backend.suffixes[0]
    assert backend.supports_statements


def test_option_limit_and_configuration_errors():
    backend = fake_backend()
    with pytest.raises(BackendError, match="at most 26 options"):
        backend.score_batch(
            [DecisionRequest(state="s", question="q", choices=[f"o{i}" for i in range(27)])]
        )
    for kwargs in (
        {"permutations": 0},
        {"max_length": 32},
        {"template": "bad"},
        {"model_path": "x"},
    ):
        with pytest.raises(BackendError):
            DecoderBackend(**backend_options("decoder"), **kwargs)
    assert backend.score_batch([]) == [] and backend.score_statements([]) == []


def test_state_prefix_is_truncated_but_question_and_options_survive():
    backend = fake_backend(max_length=128)
    long_state = " ".join(f"w{i}" for i in range(500))
    with pytest.warns(UserWarning, match="State truncated for 1"):
        backend.score_batch(
            [DecisionRequest(state=long_state, question="the question", choices=["a", "best"])]
        )
    prefix = backend._tokenizer.decode(backend.prefixes[0])
    assert "w0" in prefix and "w499" not in prefix
    assert "the question" in backend.suffixes[0] and "B. best" in backend.suffixes[0]
    assert backend.metadata["truncated_states"] == 1
    with pytest.raises(BackendError, match="too long"):
        backend.score_batch(
            [DecisionRequest(state="s", question=" ".join(["q"] * 100), choices=["a", "b"])]
        )


def test_registry_exposes_lazy_decoder():
    backend = create_backend("decoder")
    assert backend.name == "decoder" and backend._model is None
    assert backend.max_length == 2048
    assert create_backend("qwen").model_id == backend.model_id

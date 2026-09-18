"""Task routing and elimination over wide candidate sets."""

from __future__ import annotations

import pytest
from test_core import FixedBackend

from opendecision import DecisionModel
from opendecision.backends.decoder import MAX_OPTIONS
from opendecision.errors import BackendError
from opendecision.routing import TASKS, profiles, recommend


class CappedBackend(FixedBackend):
    """Scores by label order so the strongest candidate is predictable."""

    max_options = 4

    def score_batch(self, requests):
        self.requests.extend(requests)
        return [[float(sorted(r.labels).index(c)) for c in r.labels] for r in requests]


def test_every_profile_names_a_real_backend_and_cites_evidence():
    from opendecision.registry import list_models

    known = {model["name"] for model in list_models()}
    assert profiles(), "the table must not be empty"
    for profile in profiles():
        assert profile.backend in known, profile.task
        assert set(profile.runners_up) <= known, profile.task
        assert profile.backend not in profile.runners_up
        # A recommendation without a number is an opinion, not a measurement.
        assert any(character.isdigit() for character in profile.evidence), profile.task
        assert profile.task in profile.describe()
    assert [p.task for p in profiles()] == sorted(TASKS)


def test_recommend_returns_the_measured_choice_and_rejects_unknown_tasks():
    assert recommend("verification").backend == "qwen35"
    assert recommend("relevance").backend == "multilingual"
    assert recommend("stated_facts").backend == "decoder"
    assert recommend("rubric").backend == "qwen35_4b"
    with pytest.raises(BackendError, match="Unknown task"):
        recommend("chess")


def test_apple_only_recommendations_name_a_portable_alternative():
    """MLX runs on Apple silicon only, so those entries must not strand other platforms."""
    from opendecision.backends.catalog import MODEL_SPECS

    mlx = {name for name, spec in MODEL_SPECS.items() if spec["family"] == "decoder-mlx"}
    assert mlx, "the catalog should still hold MLX candidates"
    for profile in profiles():
        if profile.backend in mlx:
            assert profile.portable, f"{profile.task} recommends MLX with no fallback"
            assert profile.portable not in mlx, profile.task
            assert "Off Apple silicon" in profile.describe()
        # portable_backend always resolves to something runnable anywhere.
        assert profile.portable_backend not in mlx or profile.portable is None


def test_task_selects_the_backend_without_naming_it():
    model = DecisionModel(task="relevance", backend=FixedBackend())
    assert model.task == "relevance"
    assert DecisionModel(backend=FixedBackend()).task is None
    with pytest.raises(ValueError, match="either name or task"):
        DecisionModel("base", task="relevance", backend=FixedBackend())


def test_wide_choice_eliminates_down_to_one_round():
    backend = CappedBackend()
    model = DecisionModel(backend=backend)
    assert model.max_choices == 4
    choices = [f"option-{i:02d}" for i in range(19)]
    result = model.choose_wide(state="s", question="q?", choices=choices)
    # Highest-sorting label wins every group it is in, so it must win overall.
    assert result.choice == "option-18"
    assert result.metadata["candidates_considered"] == 19
    assert result.metadata["elimination_rounds"] >= 1
    assert result.metadata["finalists"] <= 4
    assert len(result.probabilities) == result.metadata["finalists"]
    assert "finalists only" in result.metadata["distribution_scope"]
    # No request ever exceeded the backend's cap.
    assert max(len(r.choices) for r in backend.requests) <= 4


def test_wide_choice_is_plain_choose_when_everything_fits():
    model = DecisionModel(backend=CappedBackend())
    result = model.choose_wide(state="s", question="q?", choices=["a", "b", "c"])
    assert result.metadata["elimination_rounds"] == 0
    assert set(result.probabilities) == {"a", "b", "c"}


def test_wide_choice_honours_thresholds_and_option_descriptions():
    model = DecisionModel(backend=CappedBackend())
    choices = [{"label": f"o{i}", "description": f"candidate {i}"} for i in range(9)]
    result = model.choose_wide(state="s", question="q?", choices=choices, margin_threshold=1.0)
    assert result.abstained and result.choice is None
    assert all(label.startswith("o") for label in result.probabilities)


@pytest.mark.parametrize("shortlist", [0, 1])
def test_wide_choice_rejects_an_unusable_shortlist(shortlist):
    model = DecisionModel(backend=CappedBackend())
    with pytest.raises(ValueError, match="at least two"):
        model.choose_wide(state="s", question="q?", choices=["a", "b"], shortlist=shortlist)


def test_backends_advertise_their_option_cap():
    from opendecision.registry import create_backend

    assert create_backend("decoder").max_options == MAX_OPTIONS == 26
    # Cross-encoders have no special cap; the schema bound of 256 applies.
    assert DecisionModel(backend=FixedBackend()).max_choices == 256

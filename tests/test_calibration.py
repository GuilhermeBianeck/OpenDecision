import pytest
from test_core import FixedBackend

from opendecision import CalibrationProfile, DecisionModel, fit_temperature
from opendecision.errors import CalibrationError


def fit(**kwargs):
    return fit_temperature(
        [[5, 0], [0, 5], [5, 0], [0, 5]],
        [0, 1, 1, 0],
        backend="fixture",
        model="fixture-v1",
        revision="abc123",
        **kwargs,
    )


def test_scaling_improves_overconfident_calibration_set_and_roundtrips(tmp_path):
    profile = fit()
    assert profile.temperature > 1
    assert profile.nll_after < profile.nll_before
    assert sum(profile.transform([5, 0])) == pytest.approx(1)
    profile.save(tmp_path / "profile.json")
    assert CalibrationProfile.load(tmp_path / "profile.json") == profile


def test_calibration_is_explicit_and_preserves_normalized_scores():
    model = DecisionModel(backend=FixedBackend(), calibration=fit())
    result = model.choose(state="", question="?", choices=["a", "b"])
    assert result.calibrated_probabilities == result.probabilities
    assert result.normalized_probabilities["a"] > result.probabilities["a"]
    assert result.metadata["calibrated"]


def test_profile_records_fitted_choice_counts_and_flags_unseen_sizes():
    profile = fit()
    assert (profile.choice_count_min, profile.choice_count_max) == (2, 2)
    assert profile.covers(2) is True
    assert profile.covers(3) is False
    legacy = profile.model_copy(update={"choice_count_min": None, "choice_count_max": None})
    assert legacy.covers(2) is None
    model = DecisionModel(backend=FixedBackend([3.0, 1.0, 0.5]), calibration=profile)
    inside = DecisionModel(backend=FixedBackend(), calibration=profile)
    assert inside.choose(state="", question="?", choices=["a", "b"]).metadata[
        "calibration_covers_choice_count"
    ]
    outside = model.choose(state="", question="?", choices=["a", "b", "c"])
    assert outside.metadata["calibration_covers_choice_count"] is False
    assert outside.metadata["calibrated"]
    uncalibrated = DecisionModel(backend=FixedBackend()).choose(
        state="", question="?", choices=["a", "b"]
    )
    assert uncalibrated.metadata["calibration_covers_choice_count"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "wrong"),
        ("revision", "wrong"),
        ("template", "short"),
        ("max_length", 256),
        ("precision", "float16"),
    ],
)
def test_profile_identity_must_match(field, value):
    with pytest.raises(CalibrationError, match=field):
        DecisionModel(backend=FixedBackend(), calibration=fit().model_copy(update={field: value}))


@pytest.mark.parametrize(
    "rows,labels",
    [([], []), ([[1, 0]], []), ([[1, 0]], [2]), ([[1, 0]], [True]), ([[float("nan"), 0]], [0])],
)
def test_bad_calibration_data(rows, labels):
    with pytest.raises(CalibrationError):
        fit_temperature(rows, labels, backend="x", model="x", revision="x")


def test_missing_profile_is_actionable(tmp_path):
    with pytest.raises(CalibrationError, match="Cannot load"):
        CalibrationProfile.load(tmp_path / "missing.json")

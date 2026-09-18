import pytest
from pydantic import ValidationError
from test_core import FixedBackend

from opendecision import CalibrationProfile, DecisionModel, fit_temperature
from opendecision.confidence import softmax
from opendecision.errors import CalibrationError


def fit(**kwargs):
    kwargs.setdefault("per_choice_count", False)
    return fit_temperature(
        [[5, 0], [0, 5], [5, 0], [0, 5]],
        [0, 1, 1, 0],
        backend="fixture",
        model="fixture-v1",
        revision="abc123",
        **kwargs,
    )


def mixed_rows(binary=40, quad=40):
    """Overconfident binary rows and underconfident four-way rows in one dataset."""
    rows, labels = [], []
    for index in range(binary):
        rows.append([9.0, 0.0] if index % 2 == 0 else [0.0, 9.0])
        # A third of the confident binary rows are wrong, so they need smoothing.
        labels.append((index % 2) if index % 3 else (1 - index % 2))
    for index in range(quad):
        target = index % 4
        row = [0.0] * 4
        row[target] = 0.4
        rows.append(row)
        labels.append(target)
    return rows, labels


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


def test_per_count_temperatures_beat_one_pooled_temperature():
    rows, labels = mixed_rows()
    shared = dict(backend="fixture", model="fixture-v1", revision="abc123")
    pooled = fit_temperature(rows, labels, per_choice_count=False, **shared)
    per_count = fit_temperature(rows, labels, **shared)
    assert pooled.temperatures is None
    assert set(per_count.temperatures) == {"2", "4"}
    # Confident-but-wrong binary rows need smoothing; flat four-way rows need sharpening.
    assert per_count.temperature_for(2) > 1 > per_count.temperature_for(4)
    assert per_count.nll_after < pooled.nll_after < pooled.nll_before
    assert per_count.covers(2) and per_count.covers(4)
    assert per_count.covers(3) is False
    # An unfitted count still gets the pooled fallback rather than an error.
    assert per_count.temperature_for(3) == pytest.approx(per_count.temperature)


def test_counts_below_the_minimum_fall_back_to_pooled():
    rows, labels = mixed_rows(binary=40, quad=6)
    profile = fit_temperature(
        rows, labels, backend="f", model="f", revision="r", min_rows_per_count=25
    )
    assert set(profile.temperatures) == {"2"}
    assert profile.temperature_for(4) == pytest.approx(profile.temperature)
    assert profile.covers(4) is False
    with pytest.raises(CalibrationError, match="min_rows_per_count"):
        fit_temperature(rows, labels, backend="f", model="f", revision="r", min_rows_per_count=1)


def test_transform_and_results_use_the_count_specific_temperature():
    rows, labels = mixed_rows()
    profile = fit_temperature(
        rows, labels, backend="fixture", model="fixture-v1", revision="abc123"
    )
    binary = profile.transform([1.0, 0.0])
    quad = profile.transform([1.0, 0.0, 0.0, 0.0])
    assert binary == pytest.approx(softmax([1.0, 0.0], profile.temperature_for(2)))
    assert quad == pytest.approx(softmax([1.0, 0.0, 0.0, 0.0], profile.temperature_for(4)))
    model = DecisionModel(backend=FixedBackend([1.0, 0.0]), calibration=profile)
    result = model.choose(state="", question="?", choices=["a", "b"])
    assert result.metadata["calibration_temperature"] == pytest.approx(profile.temperature_for(2))
    assert result.metadata["calibration_covers_choice_count"] is True
    plain = DecisionModel(backend=FixedBackend()).choose(state="", question="?", choices=["a", "b"])
    assert plain.metadata["calibration_temperature"] is None


@pytest.mark.parametrize("temperatures", [{"1": 2.0}, {"a": 2.0}, {"2": 0.0}, {"2": -1.0}])
def test_invalid_temperature_maps_are_rejected(temperatures):
    with pytest.raises(ValidationError):
        fit().model_copy(update={"temperatures": temperatures}).model_validate(
            {**fit().model_dump(), "temperatures": temperatures}
        )


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

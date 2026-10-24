"""Shipped calibration profiles apply only where their identity matches."""

from __future__ import annotations

from opendecision import DecisionModel
from opendecision.calibration import BUNDLED_PROFILES, CalibrationProfile
from opendecision.cli import _profile


def test_lightning_ships_a_profile_and_uses_it_by_default():
    profile = CalibrationProfile.load(BUNDLED_PROFILES / "lightning.json")
    assert profile.backend == "lightning"
    model = DecisionModel("lightning")
    assert model.calibration is not None
    assert model.calibration.temperatures == profile.temperatures


def test_a_different_context_or_an_explicit_opt_out_runs_uncalibrated():
    assert DecisionModel("lightning", max_length=2048).calibration is None
    assert DecisionModel("lightning", calibration=False).calibration is None


def test_backends_without_a_shipped_profile_are_unchanged():
    assert DecisionModel("demo").calibration is None


def test_cli_profile_values():
    assert _profile(None) is None
    assert _profile("none") is False


def test_profiles_record_the_order_averaging_they_were_fitted_on():
    from opendecision.calibration import fit_temperature

    rows, labels = [[2.0, 0.0], [0.0, 1.0], [1.0, 0.5]] * 10, [0, 1, 0] * 10
    profile = fit_temperature(rows, labels, backend="b", model="m", revision="r", order_averaging=2)
    assert profile.order_averaging == 2
    assert fit_temperature(rows, labels, backend="b", model="m", revision="r").order_averaging == 1


def test_a_profile_fitted_without_averaging_is_refused_with_it():
    import pytest

    from opendecision.errors import CalibrationError

    profile = CalibrationProfile.load(BUNDLED_PROFILES / "lightning.json")
    assert profile.order_averaging == 1  # older profiles default to single-order
    with pytest.raises(CalibrationError, match="order_averaging"):
        DecisionModel("lightning", permutations=2, calibration=profile)


def test_averaging_looks_for_its_own_shipped_profile(tmp_path, monkeypatch):
    from opendecision import calibration

    profile = CalibrationProfile.load(BUNDLED_PROFILES / "lightning.json")
    monkeypatch.setattr(calibration, "BUNDLED_PROFILES", tmp_path)
    profile.model_copy(update={"order_averaging": 2}).save(tmp_path / "lightning-k2.json")
    assert DecisionModel("lightning", permutations=2).calibration.order_averaging == 2
    assert DecisionModel("lightning").calibration is None  # no single-order file here

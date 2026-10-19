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

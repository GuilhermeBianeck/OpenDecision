"""Public, actionable runtime errors (never include user state)."""


class OpenDecisionError(Exception):
    """Base error for failures outside input validation."""


class BackendError(OpenDecisionError):
    """A model could not score a request."""


class ModelNotAvailableError(BackendError):
    """Weights or optional inference dependencies are unavailable locally."""


class CalibrationError(OpenDecisionError):
    """A calibration profile is invalid or incompatible with the model."""

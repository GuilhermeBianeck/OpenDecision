"""Public, actionable runtime errors (never include user state)."""


class OpenDecisionError(Exception):
    """Base error for failures outside input validation."""


class BackendError(OpenDecisionError):
    """A model could not score a request."""


class ModelNotAvailableError(BackendError):
    """Weights or optional inference dependencies are unavailable locally."""


class CalibrationError(OpenDecisionError):
    """A calibration profile is invalid or incompatible with the model."""


class ContextOverflow(BackendError):
    """The input is longer than the model's context and trimming it was refused."""

    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(
            f"The request needs about {tokens} tokens; the model accepts at most {limit}. "
            "Shorten the state or raise max_length within the model's context limit."
        )
        self.tokens = tokens
        self.limit = limit

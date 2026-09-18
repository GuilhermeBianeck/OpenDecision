"""OpenDecision: local finite-choice scoring with explicit uncertainty."""

from .api import DecisionModel
from .calibration import CalibrationProfile, fit_temperature
from .schemas import (
    BooleanResult,
    DecisionRequest,
    DecisionResult,
    RankedChoice,
    RankingResult,
    StatementRequest,
)

__version__ = "0.1.0a1"
__all__ = [
    "DecisionModel",
    "DecisionRequest",
    "DecisionResult",
    "BooleanResult",
    "RankedChoice",
    "RankingResult",
    "StatementRequest",
    "CalibrationProfile",
    "fit_temperature",
]

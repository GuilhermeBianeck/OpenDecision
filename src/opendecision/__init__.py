"""OpenDecision: local finite-choice scoring with explicit uncertainty."""

from .api import DecisionModel
from .calibration import CalibrationProfile, fit_temperature
from .schemas import (
    BooleanQuestion,
    BooleanResult,
    ChoiceOption,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResult,
    QuestionsRequest,
    RankedChoice,
    RankingResult,
    ScoreQuestion,
    ScoreRequest,
    ScoreResult,
    StatementRequest,
)

__version__ = "0.1.0a1"
__all__ = [
    "DecisionModel",
    "ChoiceOption",
    "DecisionRequest",
    "DecisionResult",
    "BooleanResult",
    "RankedChoice",
    "RankingResult",
    "ChoiceQuestion",
    "BooleanQuestion",
    "ScoreQuestion",
    "QuestionsRequest",
    "ScoreRequest",
    "ScoreResult",
    "StatementRequest",
    "CalibrationProfile",
    "fit_temperature",
]

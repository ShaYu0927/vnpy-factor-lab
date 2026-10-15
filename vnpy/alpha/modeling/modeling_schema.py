from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True, frozen=True)
class FactorObservation:
    """One symbol-date row before a forward-return label is attached."""
    trade_date: date
    symbol: str
    close: float
    features: dict[str, float]


@dataclass(slots=True, frozen=True)
class ModelPrediction:
    """A model score for one symbol on the prediction date."""
    trade_date: date
    symbol: str
    predicted_return: float
    rank: int


@dataclass(slots=True, frozen=True)
class RegressionMetrics:
    """Out-of-sample regression metrics."""
    mse: float
    r2: float


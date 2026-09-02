from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from .schema import FactorObservation


@dataclass(frozen=True, slots=True)
class ModelTrainingRequest:
    """Typed input of a model-training service.

    This is deliberately independent of EngineEvent.  A Python implementation
    or a future pybind11-backed C++ implementation can implement the same
    virtual interface without depending on the event engine.
    """

    observations: Sequence[FactorObservation]
    feature_names: tuple[str, ...]
    horizon: int = 5
    model_output: str | None = None
    signal_output: str | None = None
    evaluate_factors: bool = False
    factor_quantiles: int = 2
    min_abs_ic: float = 0.02
    min_abs_ic_ir: float = 0.20

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("model training observations must not be empty")
        if not all(isinstance(item, FactorObservation) for item in self.observations):
            raise TypeError("model training observations contain invalid items")
        if not self.feature_names or not all(self.feature_names):
            raise ValueError("model training feature_names must not be empty")
        if self.horizon <= 0:
            raise ValueError("model training horizon must be greater than zero")


class ModelTrainingService(ABC):
    """Polymorphic model-training boundary (C++ pure-virtual equivalent)."""

    @abstractmethod
    def train(self, request: ModelTrainingRequest) -> Any:
        """Train a model and return an implementation-specific result."""
        raise NotImplementedError


class LegacyModelTrainingService(ModelTrainingService):
    """Adapter retaining the existing factor.model_pipeline behavior."""

    def train(self, request: ModelTrainingRequest) -> Any:
        # Lazy import keeps optional ML dependencies outside module startup.
        from vnpy.factor.model_pipeline import train_and_predict_latest

        return train_and_predict_latest(
            observations=request.observations,
            feature_names=request.feature_names,
            horizon=request.horizon,
            model_output=request.model_output,
            signal_output=request.signal_output,
            evaluate_factors=request.evaluate_factors,
            factor_quantiles=request.factor_quantiles,
            min_abs_ic=request.min_abs_ic,
            min_abs_ic_ir=request.min_abs_ic_ir,
        )


def training_request_from_event(data: dict[str, Any]) -> ModelTrainingRequest:
    observations = data.get("observations")
    feature_names = data.get("feature_names")
    if not isinstance(observations, (list, tuple)):
        raise TypeError("model training observations must be a sequence")
    if not isinstance(feature_names, (list, tuple)):
        raise TypeError("model training feature_names must be a sequence")
    return ModelTrainingRequest(
        observations=observations,
        feature_names=tuple(feature_names),
        horizon=int(data.get("horizon", 5)),
        model_output=data.get("model_output"),
        signal_output=data.get("signal_output"),
        evaluate_factors=bool(data.get("evaluate_factors", False)),
        factor_quantiles=int(data.get("factor_quantiles", 2)),
        min_abs_ic=float(data.get("min_abs_ic", 0.02)),
        min_abs_ic_ir=float(data.get("min_abs_ic_ir", 0.20)),
    )

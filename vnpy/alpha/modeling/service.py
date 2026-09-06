from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import FactorObservation
from .factor_selection import AlphaFactorSelector
from .workflow import AlphaModelWorkflow


@dataclass(slots=True)
class ModelTrainingResult:
    training_samples: int
    test_samples: int
    mse: float
    r2: float
    predictions: list[tuple[str, float]]
    selected_features: tuple[str, ...]


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


class DefaultModelTrainingService(ModelTrainingService):
    """Run the canonical alpha modeling workflow."""

    def train(self, request: ModelTrainingRequest) -> Any:
        selector = None
        if request.evaluate_factors:
            selector = AlphaFactorSelector(
                min_abs_rank_ic=request.min_abs_ic,
                min_abs_rank_ic_ir=request.min_abs_ic_ir,
            )
        result = AlphaModelWorkflow(
            request.feature_names,
            request.horizon,
            factor_selector=selector,
        ).run(request.observations)
        if request.model_output:
            result.artifact.save(request.model_output)
        if request.signal_output:
            self._write_signals(result.predictions, request.signal_output)
        return ModelTrainingResult(
            training_samples=result.split.train.height + result.split.valid.height,
            test_samples=result.split.test.height,
            mse=result.metrics.mse,
            r2=result.metrics.r2,
            predictions=[
                (item.symbol, item.predicted_return)
                for item in result.predictions
            ],
            selected_features=result.artifact.feature_names,
        )

    @staticmethod
    def _write_signals(predictions: Sequence[Any], output: str) -> None:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["trade_date", "symbol", "predicted_return", "rank"])
            for item in predictions:
                writer.writerow([
                    item.trade_date.isoformat(),
                    item.symbol,
                    item.predicted_return,
                    item.rank,
                ])


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
        min_abs_ic=float(data.get("min_abs_ic", 0.02)),
        min_abs_ic_ir=float(data.get("min_abs_ic_ir", 0.20)),
    )

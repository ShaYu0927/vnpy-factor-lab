from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

from vnpy.alpha.model.template import AlphaModel
from vnpy.alpha.modeling import FactorObservation, LinearModelWorkflow, RegressionMetrics

from .config import PipelineConfig
from .experiment import LocalRecorder
from .model import ModelBundle, SignalFrame, fingerprint_observations
from .registry import ComponentRegistry


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    run_path: Path
    bundle: ModelBundle
    signals: SignalFrame
    metrics: RegressionMetrics


class LinearTrainingPipeline:
    """Compatibility workflow that composes the existing alpha modeling stack."""

    def __init__(self, config: PipelineConfig, registry: ComponentRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or default_registry()

    def run(self, observations: Sequence[FactorObservation]) -> PipelineResult:
        if not observations:
            raise ValueError("observations must not be empty")
        model = cast(AlphaModel, self.registry.create("model", self.config.model))
        recorder = cast(
            LocalRecorder,
            self.registry.create("recorder", self.config.experiment.recorder),
        )
        training = self.config.training
        fingerprint = fingerprint_observations(observations)

        with recorder.start_run(self.config.experiment.name, {"workflow": "linear_training"}) as run:
            run.log_params({
                "config": self.config.as_dict(),
                "data_fingerprint": fingerprint,
                "observation_count": len(observations),
            })
            workflow = LinearModelWorkflow(
                training.feature_names,
                horizon=training.horizon,
                model=model,
                standardize=training.standardize,
            )
            result = workflow.run(observations)
            bundle = ModelBundle.create(
                result.artifact,
                fingerprint,
                {"experiment": self.config.experiment.name},
            )
            signals = SignalFrame.from_predictions(
                result.predictions,
                bundle.model_id,
                training.horizon,
            )

            bundle_path = run.artifact_path / "model_bundle.pkl"
            signal_path = run.artifact_path / "signals.parquet"
            bundle.save(bundle_path)
            signals.write_parquet(signal_path)
            run.write_json_artifact("config.json", self.config.as_dict())
            run.write_json_artifact("data.json", {
                "fingerprint": fingerprint,
                "observation_count": len(observations),
            })
            run.log_metrics({"mse": result.metrics.mse, "r2": result.metrics.r2})

            return PipelineResult(run.run_id, run.path, bundle, signals, result.metrics)


def default_registry() -> ComponentRegistry:
    from vnpy.alpha.model.models.linear_regression_model import LinearRegressionModel

    registry = ComponentRegistry()
    registry.register("model", "linear", LinearRegressionModel)
    # Optional alpha dependencies are imported only when the registry is built.
    # A minimal installation can still use config, schemas, and recording.
    try:
        from vnpy.alpha.model.models.lasso_model import LassoModel
        registry.register("model", "lasso", LassoModel)
    except ImportError:
        pass
    try:
        from vnpy.alpha.model.models.lgb_model import LgbModel
        registry.register("model", "lgb", LgbModel)
    except ImportError:
        pass
    try:
        from vnpy.alpha.model.models.mlp_model import MlpModel
        registry.register("model", "mlp", MlpModel)
    except ImportError:
        pass
    registry.register("recorder", "local", LocalRecorder)
    return registry


def load_observations(path: str | Path) -> list[FactorObservation]:
    """Load a small interoperable JSON input used by the initial CLI slice."""
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("observation file must contain a JSON list")
    observations: list[FactorObservation] = []
    from datetime import date

    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"observation {index} must be an object")
        try:
            features = {str(key): float(value) for key, value in row["features"].items()}
            observations.append(FactorObservation(
                trade_date=date.fromisoformat(row["trade_date"]),
                symbol=str(row["symbol"]),
                close=float(row["close"]),
                features=features,
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid observation at index {index}: {exc}") from exc
    return observations

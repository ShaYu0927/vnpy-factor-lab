from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Safe component description resolved through a registry."""

    type: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str) -> ComponentSpec:
        component_type = value.get("type")
        if not isinstance(component_type, str) or not component_type.strip():
            raise ValueError(f"{location}.type must be a non-empty string")
        kwargs = value.get("kwargs", {})
        if not isinstance(kwargs, Mapping):
            raise ValueError(f"{location}.kwargs must be an object")
        return cls(component_type.strip(), dict(kwargs))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    recorder: ComponentSpec


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    feature_names: tuple[str, ...]
    horizon: int = 5
    standardize: bool = True


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    experiment: ExperimentConfig
    training: TrainingConfig
    model: ComponentSpec
    source: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        source: str | Path | None = None,
    ) -> PipelineConfig:
        experiment_value = _mapping(value.get("experiment"), "experiment")
        name = experiment_value.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("experiment.name must be a non-empty string")
        recorder = ComponentSpec.from_mapping(
            _mapping(experiment_value.get("recorder", {"type": "local"}), "experiment.recorder"),
            "experiment.recorder",
        )

        training_value = _mapping(value.get("training"), "training")
        raw_features = training_value.get("feature_names")
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError("training.feature_names must be a non-empty list")
        if not all(isinstance(item, str) and item for item in raw_features):
            raise ValueError("training.feature_names must contain only non-empty strings")
        if len(set(raw_features)) != len(raw_features):
            raise ValueError("training.feature_names must not contain duplicates")
        horizon = training_value.get("horizon", 5)
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("training.horizon must be a positive integer")
        standardize = training_value.get("standardize", True)
        if not isinstance(standardize, bool):
            raise ValueError("training.standardize must be a boolean")

        model = ComponentSpec.from_mapping(
            _mapping(value.get("model"), "model"),
            "model",
        )
        source_path = Path(source).expanduser().resolve() if source is not None else None
        return cls(
            ExperimentConfig(name.strip(), recorder),
            TrainingConfig(tuple(raw_features), horizon, standardize),
            model,
            source_path,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "name": self.experiment.name,
                "recorder": {
                    "type": self.experiment.recorder.type,
                    "kwargs": dict(self.experiment.recorder.kwargs),
                },
            },
            "training": {
                "feature_names": list(self.training.feature_names),
                "horizon": self.training.horizon,
                "standardize": self.training.standardize,
            },
            "model": {"type": self.model.type, "kwargs": dict(self.model.kwargs)},
        }


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to load YAML pipeline configs") from exc
        raw = yaml.safe_load(text)
    else:
        raise ValueError("pipeline config must use .json, .yaml, or .yml")
    return PipelineConfig.from_mapping(_mapping(raw, "root"), source)


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value

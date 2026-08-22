from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vnpy.alpha.model.template import AlphaModel

from .preprocessing import StandardFeaturePipeline


@dataclass(slots=True)
class ModelArtifact:
    """Persist the model and the exact inference contract together."""

    model: AlphaModel
    preprocessor: StandardFeaturePipeline
    feature_names: tuple[str, ...]
    label_horizon: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        with Path(path).expanduser().resolve().open("rb") as file:
            artifact = pickle.load(file)
        if not isinstance(artifact, cls):
            raise TypeError("model file does not contain a ModelArtifact")
        return artifact


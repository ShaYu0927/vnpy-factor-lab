from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl


class StandardFeaturePipeline:
    """Training-fitted standardization with stable feature ordering."""

    def __init__(self, feature_names: Sequence[str], enabled: bool = True) -> None:
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.enabled: bool = enabled
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    def fit(self, frame: pl.DataFrame) -> None:
        for name in self.feature_names:
            values = frame[name].to_numpy().astype(float)
            mean = float(np.mean(values))
            scale = float(np.std(values))
            self.means[name] = mean
            self.scales[name] = scale if scale > 0 else 1.0

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.enabled:
            return frame
        if set(self.means) != set(self.feature_names):
            raise ValueError("feature pipeline is not fitted yet")
        expressions = [
            ((pl.col(name) - self.means[name]) / self.scales[name]).alias(name)
            for name in self.feature_names
        ]
        return frame.with_columns(expressions)

    def fit_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.fit(frame)
        return self.transform(frame)


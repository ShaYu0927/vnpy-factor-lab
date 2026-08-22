from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
import polars as pl


class Reweighter(metaclass=ABCMeta):
    """Generate one non-negative sample weight for each learning row."""

    @abstractmethod
    def reweight(self, frame: pl.DataFrame) -> np.ndarray:
        """Return weights in the same row order as ``frame``."""
        raise NotImplementedError

    def __call__(self, frame: pl.DataFrame) -> np.ndarray:
        return self.reweight(frame)


class ColumnReweighter(Reweighter):
    """Read sample weights from a numeric dataframe column."""

    def __init__(self, column: str = "weight") -> None:
        self.column = column

    def reweight(self, frame: pl.DataFrame) -> np.ndarray:
        if self.column not in frame.columns:
            raise ValueError(f"weight column {self.column!r} is missing")
        return frame[self.column].to_numpy().astype(float)


def validate_weights(weights: np.ndarray, sample_count: int) -> np.ndarray:
    """Validate and normalize a reweighter result for model libraries."""
    values = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != sample_count:
        raise ValueError(f"reweighter returned {len(values)} weights for {sample_count} samples")
    if not np.isfinite(values).all():
        raise ValueError("sample weights must be finite")
    if (values < 0).any():
        raise ValueError("sample weights must be non-negative")
    if not values.any():
        raise ValueError("at least one sample weight must be positive")
    return values

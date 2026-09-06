from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from .alpha_analysis import AlphaAnalyzer


@dataclass(frozen=True, slots=True)
class FactorSelectionMetric:
    """Training-period evidence for one candidate factor."""

    factor_name: str
    mean_ic: float
    ic_ir: float
    direction: int
    selected: bool
    mean_rank_ic: float = float("nan")
    rank_ic_ir: float = float("nan")
    observations: int = 0


@dataclass(frozen=True, slots=True)
class FactorSelectionResult:
    selected_features: tuple[str, ...]
    metrics: tuple[FactorSelectionMetric, ...]


class AlphaFactorSelector:
    """Select factors from the same labeled training frame used by the model."""

    def __init__(
        self,
        *,
        min_abs_rank_ic: float = 0.02,
        min_abs_rank_ic_ir: float = 0.20,
        min_assets: int = 2,
        min_observations: int = 20,
    ) -> None:
        if min_abs_rank_ic < 0 or min_abs_rank_ic_ir < 0:
            raise ValueError("factor thresholds must not be negative")
        if min_observations <= 0:
            raise ValueError("min_observations must be greater than zero")
        self.min_abs_rank_ic = min_abs_rank_ic
        self.min_abs_rank_ic_ir = min_abs_rank_ic_ir
        self.min_observations = min_observations
        self.analyzer = AlphaAnalyzer(min_assets=min_assets)

    def select(
        self,
        training: pl.DataFrame,
        feature_names: Sequence[str],
    ) -> FactorSelectionResult:
        report = self.analyzer.evaluate(training, feature_names)
        metrics: list[FactorSelectionMetric] = []
        selected: list[str] = []
        for item in report.metrics:
            finite = np.isfinite(item.mean_rank_ic) and np.isfinite(item.rank_ic_ir)
            accepted = bool(
                finite
                and item.observations >= self.min_observations
                and abs(item.mean_rank_ic) >= self.min_abs_rank_ic
                and abs(item.rank_ic_ir) >= self.min_abs_rank_ic_ir
            )
            metric = FactorSelectionMetric(
                factor_name=item.factor_name,
                mean_ic=item.mean_ic,
                ic_ir=item.ic_ir,
                direction=1 if item.mean_rank_ic >= 0 else -1,
                selected=accepted,
                mean_rank_ic=item.mean_rank_ic,
                rank_ic_ir=item.rank_ic_ir,
                observations=item.observations,
            )
            metrics.append(metric)
            if accepted:
                selected.append(item.factor_name)
        if not selected:
            raise ValueError("no factor passed the training-period Rank IC thresholds")
        return FactorSelectionResult(tuple(selected), tuple(metrics))


# Compatibility name; new code should use AlphaFactorSelector.
AlphalensFactorSelector = AlphaFactorSelector

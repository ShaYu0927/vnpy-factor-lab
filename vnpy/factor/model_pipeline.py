"""Compatibility facade for the alpha modeling workflow.

Factor code may keep importing this module while model-specific responsibilities
live under :mod:`vnpy.alpha.modeling`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from vnpy.alpha.modeling import (
    AlphalensFactorSelector,
    FactorObservation,
    ForwardReturnDatasetBuilder,
    LinearModelWorkflow,
)
from vnpy.factor.core.factor_engine import FactorBatchResult


@dataclass(slots=True)
class BatchModelResult:
    training_samples: int
    test_samples: int
    mse: float
    r2: float
    predictions: list[tuple[str, float]]
    selected_features: tuple[str, ...]


def collect_observations(result: FactorBatchResult, closes: dict[str, float], feature_names: Sequence[str], trade_date: date) -> list[FactorObservation]:
    """Convert long-form factor results into complete model feature rows."""
    observations: list[FactorObservation] = []
    for symbol, close in closes.items():
        values = result.scalar_map(symbol)
        if not all(name in values and values[name] is not None for name in feature_names):
            continue
        features = {name: float(values[name]) for name in feature_names}  # type: ignore[arg-type]
        if not np.isfinite(list(features.values())).all() or close <= 0:
            continue
        observations.append(FactorObservation(trade_date, symbol, close, features))
    return observations


def build_labeled_frame(
    observations: Sequence[FactorObservation],
    feature_names: Sequence[str],
    horizon: int,
) -> pl.DataFrame:
    """Backward-compatible label builder."""
    return ForwardReturnDatasetBuilder(feature_names, horizon).build(observations)


def train_and_predict_latest(
    observations: Sequence[FactorObservation],
    feature_names: Sequence[str],
    horizon: int = 5,
    model_output: str | None = None,
    signal_output: str | None = None,
    evaluate_factors: bool = False,
    factor_quantiles: int = 2,
    min_abs_ic: float = 0.02,
    min_abs_ic_ir: float = 0.20,
) -> BatchModelResult:
    """Run the modeling workflow and retain the existing batch API."""
    selector = None
    if evaluate_factors:
        selector = AlphalensFactorSelector(
            horizon=horizon,
            quantiles=factor_quantiles,
            min_abs_ic=min_abs_ic,
            min_abs_ic_ir=min_abs_ic_ir,
        )
    result = LinearModelWorkflow(
        feature_names,
        horizon,
        factor_selector=selector,
    ).run(observations)
    predictions = [
        (prediction.symbol, prediction.predicted_return)
        for prediction in result.predictions
    ]

    if model_output:
        if result.factor_selection is not None:
            result.artifact.metadata["factor_selection"] = [
                {
                    "factor_name": metric.factor_name,
                    "mean_ic": metric.mean_ic,
                    "ic_ir": metric.ic_ir,
                    "direction": metric.direction,
                    "selected": metric.selected,
                }
                for metric in result.factor_selection.metrics
            ]
        result.artifact.save(model_output)
    if signal_output:
        path = Path(signal_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["trade_date", "symbol", "predicted_return", "rank"])
            for prediction in result.predictions:
                writer.writerow([
                    prediction.trade_date.isoformat(),
                    prediction.symbol,
                    prediction.predicted_return,
                    prediction.rank,
                ])

    training_samples = result.split.train.height + result.split.valid.height
    if result.factor_selection is not None:
        print("\n训练期 IC 检验", flush=True)
        print(
            f"{'因子':<26} {'平均IC':>10} {'ICIR':>10} "
            f"{'方向':>8} {'入选':>8}",
            flush=True,
        )
        for metric in result.factor_selection.metrics:
            print(
                f"{metric.factor_name:<28} {metric.mean_ic:>10.6f} "
                f"{metric.ic_ir:>10.6f} {metric.direction:>8} "
                f"{('是' if metric.selected else '否'):>8}",
                flush=True,
            )

    return BatchModelResult(
        training_samples=training_samples,
        test_samples=result.split.test.height,
        mse=result.metrics.mse,
        r2=result.metrics.r2,
        predictions=predictions,
        selected_features=result.artifact.feature_names,
    )

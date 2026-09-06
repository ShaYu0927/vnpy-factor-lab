from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True, slots=True)
class AlphaMetric:
    """Cross-sectional signal statistics for one alpha feature."""

    factor_name: str
    mean_ic: float
    ic_ir: float
    mean_rank_ic: float
    rank_ic_ir: float
    observations: int


@dataclass(frozen=True, slots=True)
class AlphaAnalysisReport:
    """Daily IC series and their time-series summaries."""

    daily: pl.DataFrame
    metrics: tuple[AlphaMetric, ...]


class AlphaDatasetBuilder:
    """Join alpha features to an explicitly timed forward-return label.

    A feature observed after the close on T can first be traded at T+1.  With
    the defaults, the label is close(T+2) / close(T+1) - 1, matching Qlib's
    daily Alpha158 convention.  Set entry_offset=0 when the feature is known
    and executable at the T close.
    """

    def __init__(self, horizon: int = 1, entry_offset: int = 1) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        if entry_offset < 0:
            raise ValueError("entry_offset must not be negative")
        self.horizon = horizon
        self.entry_offset = entry_offset

    def build(
        self,
        features: pl.DataFrame,
        prices: pl.DataFrame,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        keys = ["datetime", "vt_symbol"]
        self._validate_frame(features, keys, "features")
        self._validate_frame(prices, [*keys, "close"], "prices")

        names = tuple(feature_names or (name for name in features.columns if name not in keys))
        if not names:
            raise ValueError("features must contain at least one alpha column")
        missing = set(names) - set(features.columns)
        if missing:
            raise ValueError(f"features are missing columns: {', '.join(sorted(missing))}")

        price_labels = (
            prices.select([*keys, "close"])
            .sort(["vt_symbol", "datetime"])
            .with_columns(
                pl.col("close").shift(-self.entry_offset).over("vt_symbol").alias("_entry"),
                pl.col("close")
                .shift(-(self.entry_offset + self.horizon))
                .over("vt_symbol")
                .alias("_exit"),
            )
            .with_columns((pl.col("_exit") / pl.col("_entry") - 1.0).alias("label"))
            .select([*keys, "label"])
        )
        return (
            features.select([*keys, *names])
            .join(price_labels, on=keys, how="inner")
            .filter(pl.col("label").is_finite())
            .sort(keys)
        )

    @staticmethod
    def _validate_frame(frame: pl.DataFrame, required: Sequence[str], name: str) -> None:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} are missing columns: {', '.join(sorted(missing))}")
        duplicates = frame.group_by(["datetime", "vt_symbol"]).len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError(f"{name} contain duplicate datetime/symbol rows")


class AlphaAnalyzer:
    """Evaluate alpha columns against an existing label by date and symbol."""

    def __init__(self, min_assets: int = 2) -> None:
        if min_assets < 2:
            raise ValueError("min_assets must be at least 2")
        self.min_assets = min_assets

    def evaluate(
        self,
        dataset: pl.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> AlphaAnalysisReport:
        required = {"datetime", "vt_symbol", "label"}
        missing = required - set(dataset.columns)
        if missing:
            raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
        names = tuple(feature_names or (name for name in dataset.columns if name not in required))
        if not names:
            raise ValueError("dataset must contain at least one alpha column")

        frame = dataset.select(["datetime", "vt_symbol", *names, "label"]).to_pandas()
        rows: list[dict[str, object]] = []
        for current_date, group in frame.groupby("datetime", sort=True):
            for name in names:
                valid = group[[name, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < self.min_assets:
                    continue
                if valid[name].nunique() < 2 or valid["label"].nunique() < 2:
                    continue
                ic = valid[name].corr(valid["label"], method="pearson")
                rank_ic = valid[name].corr(valid["label"], method="spearman")
                if pd.notna(ic) or pd.notna(rank_ic):
                    rows.append({
                        "datetime": current_date,
                        "factor_name": name,
                        "ic": float(ic),
                        "rank_ic": float(rank_ic),
                        "asset_count": len(valid),
                    })

        daily_schema = {
            "datetime": dataset.schema["datetime"],
            "factor_name": pl.String,
            "ic": pl.Float64,
            "rank_ic": pl.Float64,
            "asset_count": pl.Int64,
        }
        if rows:
            daily = pl.from_pandas(pd.DataFrame(rows)).with_columns(
                pl.col("datetime").cast(dataset.schema["datetime"]),
                pl.col("asset_count").cast(pl.Int64),
            )
        else:
            daily = pl.DataFrame(schema=daily_schema)
        metrics: list[AlphaMetric] = []
        for name in names:
            values = daily.filter(pl.col("factor_name") == name)
            ic = values["ic"].to_numpy()
            rank_ic = values["rank_ic"].to_numpy()
            metrics.append(AlphaMetric(
                factor_name=name,
                mean_ic=self._mean(ic),
                ic_ir=self._ir(ic),
                mean_rank_ic=self._mean(rank_ic),
                rank_ic_ir=self._ir(rank_ic),
                observations=len(values),
            ))
        return AlphaAnalysisReport(daily=daily, metrics=tuple(metrics))

    @staticmethod
    def _mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if len(finite) else float("nan")

    @classmethod
    def _ir(cls, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            return float("nan")
        std = finite.std(ddof=1)
        return cls._mean(finite) / float(std) if std else float("nan")

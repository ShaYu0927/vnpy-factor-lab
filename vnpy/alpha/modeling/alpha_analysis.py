from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True, slots=True)
class AlphaMetric:
    """保存单个 Alpha 因子的汇总评价指标"""

    factor_name: str     # 因子名称
    mean_ic: float       # 每日 Pearson IC 的平均值
    ic_ir: float         # IC 均值与标准差的比值。
    mean_rank_ic: float  # 每日 Spearman RankIC 的平均值
    rank_ic_ir: float    # RankIC 均值与标准差的比值
    observations: int    # 参与汇总的有效交易日数量，不是股票数量


@dataclass(frozen=True, slots=True)
class AlphaAnalysisReport:
    """保存因子评价的完整结果"""

    daily: pl.DataFrame                # 每个因子在每个交易日上的 IC、RankIC 和股票数量
    metrics: tuple[AlphaMetric, ...]   # 每个因子的时间序列汇总指标


class AlphaDatasetBuilder:
    """
    将因子值与按照交易时间对齐的未来收益 Label 合并。

    假设因子在 T 日收盘后才能得到，则最早只能在 T+1 日交易。

    默认参数：

        horizon=1
        entry_offset=1

    对应的收益标签为：

        close(T+2) / close(T+1) - 1

    如果因子在 T 日收盘前已经能够获得，并且能够按照 T 日收盘价成交，可以将 entry_offset 设置为 0
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
        """
        构建包含因子值和未来收益标签的研究数据集

        Args:
            features: AlphaEngine计算得到的因子数据 必须包含 datetime和vt_symbol。
            prices: 用于生成未来收益的价格数据       必须包含 datetime、vt_symbol和close。
            feature_names: 指定需要使用的因子列。为空时自动使用features中除主键之外的所有列。

        Returns:
            包含datetime、vt_symbol、因子列和label的数据集。

        Raises:
            ValueError: 输入缺少必要字段、存在重复主键或没有因子列。
        """
        
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
        """
        检查输入数据的字段和联合主键。

        Args:
            frame: 需要检查的DataFrame。
            required: 必须存在的字段。
            name: 数据名称，用于生成异常信息。

        Raises:
            ValueError: 缺少必要字段，或者同一股票在同一时间出现多条记录。
        """
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} are missing columns: {', '.join(sorted(missing))}")
        duplicates = frame.group_by(["datetime", "vt_symbol"]).len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError(f"{name} contain duplicate datetime/symbol rows")


class AlphaAnalyzer:
    """
    按交易日评价各个Alpha因子与未来收益之间的相关性。

    对于每个交易日和每个因子：

    1. 取出当天所有股票的因子值与未来收益；
    2. 计算Pearson IC
    3. 计算Spearman RankIC
    4. 将每日IC序列汇总为均值和ICIR。

    这里计算的是“每日横截面相关性”，不是同一只股票沿时间方向
    的相关性。
    """

    def __init__(self, min_assets: int = 2) -> None:
        if min_assets < 2:
            raise ValueError("min_assets must be at least 2")
        self.min_assets = min_assets

    def evaluate(
        self,
        dataset: pl.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> AlphaAnalysisReport:
        
        """
        评价数据集中的一个或多个Alpha因子。

        Args:
            dataset: AlphaDatasetBuilder生成的数据集 必须包含 datetime、vt_symbol、label和至少一个因子列。
            feature_names: 指定需要评价的因子。为空时自动使用 主键和label之外的所有列。
        Returns:
            包含每日IC序列和每个因子汇总指标的分析报告。

        Raises:
            ValueError: 数据集缺少必要字段或没有可评价的因子。
        """
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

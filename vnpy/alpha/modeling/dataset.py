from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import polars as pl

from vnpy.alpha.dataset import Segment

from .schema import FactorObservation


class FrameDataset:
    """Adapt prepared frames to the existing AlphaModel dataset interface."""

    def __init__(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame) -> None:
        self.frames: dict[Segment, pl.DataFrame] = {
            Segment.TRAIN: train,
            Segment.VALID: valid,
            Segment.TEST: test,
        }

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        return self.frames[segment]

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        return self.frames[segment]


@dataclass(slots=True, frozen=True)
class DatasetSplit:
    train: pl.DataFrame     # 模型通过这部分历史数据学习因子与未来收益之间的关系
    valid: pl.DataFrame     # 用于调整模型参数、比较不同模型
    test: pl.DataFrame      # 测试集


class ForwardReturnDatasetBuilder:
    """Build a model frame using a forward return measured in bars."""

    def __init__(self, feature_names: Sequence[str], horizon: int = 5) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.horizon: int = horizon

    # 根据因子观测数据构造模型训练 DataFrame
    def build(self, observations: Sequence[FactorObservation]) -> pl.DataFrame:
        by_symbol: dict[str, list[FactorObservation]] = {}
        for observation in observations:
            by_symbol.setdefault(observation.symbol, []).append(observation)

        rows: list[dict[str, object]] = []
        for symbol_observations in by_symbol.values():
            symbol_observations.sort(key=lambda item: item.trade_date)
            for index in range(len(symbol_observations) - self.horizon):
                current = symbol_observations[index]
                future = symbol_observations[index + self.horizon]
                rows.append({
                    "datetime": datetime.combine(current.trade_date, datetime.min.time()),
                    "vt_symbol": current.symbol,
                    **current.features,
                    "label": future.close / current.close - 1.0,
                })

        columns = ["datetime", "vt_symbol", *self.feature_names, "label"]
        return pl.DataFrame(rows).select(columns).sort(["datetime", "vt_symbol"])


class ChronologicalSplitter:
    """Split complete dates and purge boundary labels that look into the future."""

    def __init__(
        self,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        purge_horizon: int = 0,
    ) -> None:
        if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
            raise ValueError("split ratios must be positive and leave room for test data")
        if purge_horizon < 0:
            raise ValueError("purge_horizon must not be negative")
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.purge_horizon = purge_horizon

    def split(self, frame: pl.DataFrame) -> DatasetSplit:
        # 获取所有交易日期，并按时间排序
        dates = frame["datetime"].unique().sort().to_list()
        if len(dates) < 3:
            raise ValueError("not enough labeled dates to split train/valid/test data")

        if self.purge_horizon:
            usable_dates = len(dates) - 2 * self.purge_horizon
            if usable_dates < 3:
                raise ValueError(
                    "not enough labeled dates after purging train/valid/test boundaries"
                )
            train_count = max(1, int(usable_dates * self.train_ratio))
            valid_count = max(1, int(usable_dates * self.valid_ratio))
            if train_count + valid_count >= usable_dates:
                valid_count = max(1, usable_dates - train_count - 1)

            valid_start = train_count + self.purge_horizon
            test_start = valid_start + valid_count + self.purge_horizon
            train = frame.filter(pl.col("datetime") <= dates[train_count - 1])
            valid = frame.filter(
                (pl.col("datetime") >= dates[valid_start])
                & (pl.col("datetime") <= dates[valid_start + valid_count - 1])
            )
            test = frame.filter(pl.col("datetime") >= dates[test_start])
        else:
            train_end = max(1, int(len(dates) * self.train_ratio))
            valid_end = max(
                train_end + 1,
                int(len(dates) * (self.train_ratio + self.valid_ratio)),
            )
            valid_end = min(valid_end, len(dates) - 1)
            train = frame.filter(pl.col("datetime") <= dates[train_end - 1])
            valid = frame.filter(
                (pl.col("datetime") > dates[train_end - 1])
                & (pl.col("datetime") <= dates[valid_end - 1])
            )
            test = frame.filter(pl.col("datetime") > dates[valid_end - 1])
        return DatasetSplit(train, valid, test)

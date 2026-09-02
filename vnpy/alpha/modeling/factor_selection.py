from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from vnpy.common.logger import get_logger

from .alphalens_evaluator import AlphalensEvaluator
from .schema import FactorObservation


logger = get_logger("alpha.factor_selection")


@dataclass(frozen=True, slots=True)
class FactorSelectionMetric:
    """Training-period Alphalens statistics for one candidate factor."""
    factor_name: str    # 因子名称，例如 momentum、growth
    mean_ic: float      # 平均信息系数 IC，衡量因子预测未来收益的有效性
    ic_ir: float        # IC 信息比率，衡量因子预测能力的稳定性
    direction: int      # 因子方向，1 表示正向因子，-1 表示反向因子
    selected: bool      # 是否通过筛选条件，作为模型输入因子


@dataclass(frozen=True, slots=True)
class FactorSelectionResult:
    """Selected model features and the evidence used to select them."""

    selected_features: tuple[str, ...]          # 最终筛选通过的因子名称，用于模型训练输入
    metrics: tuple[FactorSelectionMetric, ...]  # 所有候选因子的评估结果（IC、ICIR、方向、是否选中）


class AlphalensFactorSelector:
    """Select factors using only observations from the model training period."""

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: int = 2,
        min_abs_ic: float = 0.02,
        min_abs_ic_ir: float = 0.20,
        max_loss: float = 0.50,
    ) -> None:
        self.horizon = horizon
        self.quantiles = quantiles
        self.min_abs_ic = min_abs_ic
        self.min_abs_ic_ir = min_abs_ic_ir
        self.evaluator = AlphalensEvaluator(
            periods=(horizon,),
            quantiles=quantiles,
            max_loss=max_loss,
        )

    def select(
        self,
        observations: Sequence[FactorObservation],
        feature_names: Sequence[str],
        train_end: datetime,
    ) -> FactorSelectionResult:
        training = [
            item
            for item in observations
            if item.trade_date <= train_end.date()
        ]
        asset_count = len({item.symbol for item in training})
        if asset_count < 20:
            logger.warning("横截面仅 %s 只股票,IC 结果只适合流程演示，不具备统计可靠性", asset_count)
        prices = pd.DataFrame([
            {
                "datetime": item.trade_date,
                "vt_symbol": item.symbol,
                "close": item.close,
            }
            for item in observations
        ])
        metrics: list[FactorSelectionMetric] = []
        selected: list[str] = []

        for name in feature_names:
            factor = pl.DataFrame([
                {
                    "datetime": item.trade_date,
                    "vt_symbol": item.symbol,
                    "factor": item.features[name],
                }
                for item in training
                if name in item.features
            ])
            report = self.evaluator.evaluate(factor, prices)
            period = f"{self.horizon}D"
            mean_ic = float(report.information_coefficient[period].mean())
            ic_ir = float(report.information_ratio[period])
            is_finite = np.isfinite(mean_ic) and np.isfinite(ic_ir)
            accepted = bool(
                is_finite
                and abs(mean_ic) >= self.min_abs_ic
                and abs(ic_ir) >= self.min_abs_ic_ir
            )
            metric = FactorSelectionMetric(
                factor_name=name,
                mean_ic=mean_ic,
                ic_ir=ic_ir,
                direction=1 if mean_ic >= 0 else -1,
                selected=accepted,
            )
            metrics.append(metric)
            if accepted:
                selected.append(name)

        if not selected:
            raise ValueError(
                "Alphalens rejected every candidate factor; adjust the research "
                "thresholds or provide a larger stock universe"
            )
        return FactorSelectionResult(tuple(selected), tuple(metrics))

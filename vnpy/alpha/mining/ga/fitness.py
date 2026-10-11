"""Training-only factor fitness and independent holdout scoring."""

from __future__ import annotations

import polars as pl

from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine
from vnpy.alpha.expression import Node

from .config import EvolutionConfig
from .population import Fitness


def _labels(market: pl.DataFrame, config: EvolutionConfig) -> tuple[pl.DataFrame, object]:
    """
    构造未来收益率标签，并确定训练集与留出集的分界日期。

    使用所有股票共享的交易日历定位买入、卖出日期。
    某只股票缺少行情时，不会把它的下一条行情误当成下一个交易日。
    """
    calendar = market.select("datetime").unique().sort("datetime").with_row_index("session")
    split_index = int(calendar.height * config.train_fraction)
    if split_index < config.entry_offset + config.period + config.min_days or split_index >= calendar.height:
        raise ValueError("not enough sessions for the configured training and holdout periods")
    split_date = calendar["datetime"][split_index]
    entry = calendar.select(
        (pl.col("session") - config.entry_offset).alias("session"),
        pl.col("datetime").alias("entry_date"),
    )
    exit_dates = calendar.select(
        (pl.col("session") - config.entry_offset - config.period).alias("session"),
        pl.col("datetime").alias("exit_date"),
    )
    prices = market.select("datetime", "vt_symbol", "close", "volume")
    labels = (
        market.select("datetime", "vt_symbol")
        .join(calendar, on="datetime")
        .join(entry, on="session", how="left")
        .join(exit_dates, on="session", how="left")
        .join(prices.rename({"datetime": "entry_date", "close": "entry_close", "volume": "entry_volume"}),
              on=["entry_date", "vt_symbol"], how="left")
        .join(prices.rename({"datetime": "exit_date", "close": "exit_close", "volume": "exit_volume"}),
              on=["exit_date", "vt_symbol"], how="left")
        .with_columns((pl.col("exit_close") / pl.col("entry_close") - 1).alias("label"))
        .filter(
            (pl.col("entry_close") > 0) & (pl.col("exit_close") > 0)
            & (pl.col("entry_volume") > 0) & (pl.col("exit_volume") > 0)
            & pl.col("label").is_finite()
        )
        .select("datetime", "vt_symbol", "exit_date", "label")
    )
    return labels, split_date


class FitnessEvaluator:
    """综合每日 Rank IC、覆盖率、稳定性和表达树大小, 为因子评分"""

    def __init__(self, market: pl.DataFrame, config: EvolutionConfig) -> None:
        self.market = market
        self.config = config
        self.labels, self.split_date = _labels(market, config)
        self.training_market = market.filter(pl.col("datetime") < self.split_date)
        self.training_labels = self.labels.filter(pl.col("exit_date") < self.split_date)
        self.holdout_labels = self.labels.filter(pl.col("datetime") >= self.split_date)

    def score(self, tree: Node, *, holdout: bool = False) -> Fitness:
        alpha = Alpha("fitness", tree.to_formula())
        market = self.market if holdout else self.training_market
        labels = self.holdout_labels if holdout else self.training_labels
        try:
            factors = AlphaEngine([alpha]).calculate(market)
            joined = labels.join(factors.select("datetime", "vt_symbol", "fitness"), on=["datetime", "vt_symbol"], how="left")
            finite = joined.filter(pl.col("fitness").is_finite())
            coverage = finite.height / joined.height if joined.height else 0.0
            daily = (
                finite.with_columns(
                    pl.col("fitness").rank("average").over("datetime").alias("factor_rank"),
                    pl.col("label").rank("average").over("datetime").alias("return_rank"),
                )
                .group_by("datetime")
                .agg(pl.len().alias("assets"), pl.corr("factor_rank", "return_rank").alias("ic"))
                .filter((pl.col("assets") >= self.config.min_assets) & pl.col("ic").is_finite())
            )
            days = daily.height
            ic = daily["ic"].mean() if days else None
            ic_std = daily["ic"].std() if days > 1 else None
            stability = abs(ic) / (ic_std + 1e-6) if ic is not None and ic_std is not None else None
            complexity = sum(1 for _ in tree.walk())
            score = 0.0
            if ic is not None and days >= self.config.min_days:
                stability_bonus = 1 + self.config.stability_weight * min(stability or 0.0, 1.0)
                complexity_cost = 1 + self.config.complexity_penalty * (complexity - 1)
                score = abs(ic) * coverage * stability_bonus / complexity_cost
            return Fitness(float(score), float(ic) if ic is not None else None, days, coverage,
                           ic_std=float(ic_std) if ic_std is not None else None,
                           stability=float(stability) if stability is not None else None,
                           complexity=complexity)
        except (ArithmeticError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            return Fitness(0.0, None, 0, 0.0, f"{type(exc).__name__}: {exc}",
                           complexity=sum(1 for _ in tree.walk()))

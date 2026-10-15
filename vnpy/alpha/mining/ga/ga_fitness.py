"""Training-only factor fitness and independent holdout scoring."""

from __future__ import annotations

import polars as pl

from vnpy.alpha.alpha_formula import Alpha
from vnpy.alpha.alpha_domain import MarketFrame
from vnpy.alpha.alpha_engine import AlphaEngine
from vnpy.alpha.expression import Node
from vnpy.alpha.alpha_sample_diagnostics import eligible_samples, with_history_counts
from vnpy.alpha.alpha_time_split import SessionSplit, TimeSplitConfig

from .ga_config import EvolutionConfig
from .ga_population import Fitness


def _label_frame(
    market: pl.DataFrame, config: EvolutionConfig, split: SessionSplit
) -> pl.DataFrame:
    """
    构造未来收益率标签，并确定训练集与留出集的分界日期。

    收盘因子在 T 计算，按共享交易日历取 T+entry_offset 的收盘价入场，
    再经过 period 个交易日，以收盘价退出。
    某只股票缺少行情时，不会把它的下一条行情误当成下一个交易日。
    返回原始标签，包括无未来价格的行，用于记录样本损失。
    """
    required = {"datetime", "vt_symbol", "close", "volume"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"market is missing columns: {', '.join(sorted(missing))}")
    keys = market.select("datetime", "vt_symbol")
    if any(keys.null_count().row(0)):
        raise ValueError("market keys must not be null")
    if keys.is_duplicated().any():
        raise ValueError("market contains duplicate datetime/vt_symbol rows")

    calendar = pl.DataFrame(
        {"datetime": market["datetime"].unique().sort()}
    ).with_row_index("session")
    if tuple(calendar["datetime"].to_list()) != split.sessions:
        raise ValueError("time split calendar does not match market data")
    if split.train_count < config.entry_offset + config.period + config.min_days:
        raise ValueError(
            "not enough sessions for the configured training and holdout periods"
        )
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
        .join(
            prices.rename(
                {
                    "datetime": "entry_date",
                    "close": "entry_close",
                    "volume": "entry_volume",
                }
            ),
            on=["entry_date", "vt_symbol"],
            how="left",
        )
        .join(
            prices.rename(
                {
                    "datetime": "exit_date",
                    "close": "exit_close",
                    "volume": "exit_volume",
                }
            ),
            on=["exit_date", "vt_symbol"],
            how="left",
        )
        .with_columns((pl.col("exit_close") / pl.col("entry_close") - 1).alias("label"))
        .with_columns(
            pl.when(
                (pl.col("entry_close") > 0)
                & (pl.col("exit_close") > 0)
                & (pl.col("entry_volume") > 0)
                & (pl.col("exit_volume") > 0)
                & pl.col("entry_close").is_finite()
                & pl.col("exit_close").is_finite()
                & pl.col("label").is_finite()
            )
            .then(pl.col("label"))
            .otherwise(None)
            .alias("label")
        )
        .select("datetime", "vt_symbol", "entry_date", "exit_date", "label")
    )
    return labels


def _labels(
    market: pl.DataFrame, config: EvolutionConfig
) -> tuple[pl.DataFrame, object]:
    panel = MarketFrame(market)
    split = panel.split(TimeSplitConfig(config.train_fraction))
    return _label_frame(panel.frame, config, split).filter(
        pl.col("label").is_finite()
    ), split.holdout_start


class FitnessEvaluator:
    """用有效收益标签行作覆盖率分母，按日计算 Rank IC 后汇总评分。"""

    def __init__(
        self,
        market: pl.DataFrame,
        config: EvolutionConfig,
        *,
        split: SessionSplit | None = None,
    ) -> None:
        panel = MarketFrame(market)
        self.market = panel.frame
        self.config = config
        self.split = split or panel.split(TimeSplitConfig(config.train_fraction))
        self.split_date = self.split.holdout_start
        self.raw_labels = _label_frame(self.market, config, self.split)
        self.labels = self.raw_labels.filter(pl.col("label").is_finite())
        self.history_counts = with_history_counts(self.market).select(
            "datetime", "vt_symbol", "history_bars"
        )
        self.training_market = self.market.filter(pl.col("datetime") < self.split_date)
        self.training_labels = self.labels.filter(pl.col("exit_date") < self.split_date)
        self.holdout_labels = self.labels.filter(pl.col("datetime") >= self.split_date)

    def score(self, tree: Node, *, holdout: bool = False) -> Fitness:
        """Score training by absolute IC; score holdout in the training direction."""
        complexity = sum(1 for _ in tree.walk())
        market = self.market if holdout else self.training_market
        segment = "holdout" if holdout else "train"
        labels = self.raw_labels.filter(
            pl.col("datetime") >= self.split_date
            if holdout
            else pl.col("datetime") < self.split_date,
        )
        try:
            alpha = Alpha("fitness", tree.to_formula())
            alpha_engine = AlphaEngine([alpha])
            factors = alpha_engine.calculate(market)
            joined = (
                labels.join(
                    self.history_counts, on=["datetime", "vt_symbol"], how="left"
                )
                .join(
                    factors.select("datetime", "vt_symbol", "fitness"),
                    on=["datetime", "vt_symbol"],
                    how="left",
                )
                .rename({"datetime": "date", "vt_symbol": "asset", "fitness": "factor"})
            )
            finite, sample_counts = eligible_samples(
                joined,
                input_days=len(self.split.dates_for(segment)),
                lookback=alpha_engine.min_bars,
                min_assets=self.config.min_assets,
                train_end=None if holdout else self.split_date,
            )
            denominator = sample_counts["label_eligible_rows"]
            coverage = (
                sample_counts["paired_rows"] / denominator if denominator else 0.0
            )
            daily = (
                finite.with_columns(
                    pl.col("factor").rank("average").over("date").alias("factor_rank"),
                    pl.col("label").rank("average").over("date").alias("return_rank"),
                )
                .group_by("date")
                .agg(
                    pl.len().alias("assets"),
                    pl.corr("factor_rank", "return_rank").alias("ic"),
                )
                .filter(
                    (pl.col("assets") >= self.config.min_assets)
                    & pl.col("ic").is_finite()
                )
            )
            days = daily.height
            ic = daily["ic"].mean() if days else None
            ic_std = daily["ic"].std() if days > 1 else None
            stability = (
                abs(ic) / (ic_std + 1e-6)
                if ic is not None and ic_std is not None
                else None
            )
            direction = None
            if holdout:
                training = self.score(tree)
                if training.outcome == "calculation_error":
                    return training
                direction = training.direction
            elif ic is not None and days >= self.config.min_days and ic != 0:
                direction = 1 if ic > 0 else -1

            score = 0.0
            if ic is not None and days >= self.config.min_days:
                stability_bonus = 1 + self.config.stability_weight * min(
                    stability or 0.0, 1.0
                )
                complexity_cost = 1 + self.config.complexity_penalty * (complexity - 1)
                if holdout:
                    directed_ic = ic * direction if direction is not None else 0.0
                else:
                    directed_ic = abs(ic)
                score = directed_ic * coverage * stability_bonus / complexity_cost
            outcome = (
                "scored" if days >= self.config.min_days else "insufficient_samples"
            )
            if outcome == "scored" and holdout and direction is None:
                outcome = "no_training_direction"
            return Fitness(
                float(score),
                float(ic) if ic is not None else None,
                days,
                coverage,
                ic_std=float(ic_std) if ic_std is not None else None,
                stability=float(stability) if stability is not None else None,
                complexity=complexity,
                outcome=outcome,
                direction=direction,
                sample_counts=sample_counts,
            )
        except (
            ArithmeticError,
            KeyError,
            TypeError,
            ValueError,
            pl.exceptions.PolarsError,
        ) as exc:
            return Fitness(
                0.0,
                None,
                0,
                0.0,
                f"{type(exc).__name__}: {exc}",
                complexity=complexity,
                outcome="calculation_error",
            )

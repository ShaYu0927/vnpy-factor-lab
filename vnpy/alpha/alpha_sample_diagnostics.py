"""Count sample losses in the same order for genetic fitness and factor reports."""

from __future__ import annotations

import polars as pl


def with_history_counts(market: pl.DataFrame) -> pl.DataFrame:
    return market.sort(["vt_symbol", "datetime"]).with_columns(
        pl.col("datetime").cum_count().over("vt_symbol").alias("history_bars"),
    )


def eligible_samples(
    frame: pl.DataFrame, *, input_days: int, lookback: int,
    min_assets: int, train_end=None,
) -> tuple[pl.DataFrame, dict]:
    """Input columns: date, asset, exit_date, label, factor, history_bars.

    Row losses are exclusive. Day losses count dates disappearing entirely at
    that stage; a date that loses only some stocks remains for subsequent stages.
    Constant-factor/return and insufficient-asset day counts can overlap.
    """
    current = frame
    counts = {"input_rows": frame.height, "input_days": input_days, "lookback": lookback}

    def retain(condition: pl.Expr, reason: str) -> None:
        nonlocal current
        before_rows = current.height
        before_days = current["date"].n_unique()
        current = current.filter(condition.fill_null(False))
        counts[f"{reason}_rows"] = before_rows - current.height
        counts[f"{reason}_days"] = before_days - current["date"].n_unique()

    retain(pl.col("exit_date").is_not_null(), "missing_future")
    retain(pl.col("exit_date") < train_end if train_end is not None else pl.lit(True), "boundary_dropped")
    retain(pl.col("label").is_finite(), "invalid_price")
    counts.update(label_eligible_rows=current.height, label_eligible_days=current["date"].n_unique())
    retain(pl.col("history_bars") >= lookback, "warmup")
    retain(pl.col("factor").is_finite(), "invalid_factor")
    counts["paired_rows"] = current.height
    stats = current.group_by("date").agg(
        pl.len().alias("assets"),
        pl.col("factor").n_unique().alias("factor_unique"),
        pl.col("label").n_unique().alias("return_unique"),
    )
    counts.update(
        insufficient_asset_days=stats.filter(pl.col("assets") < min_assets).height,
        constant_factor_days=stats.filter(pl.col("factor_unique") <= 1).height,
        constant_return_days=stats.filter(pl.col("return_unique") <= 1).height,
    )
    usable = stats.filter(
        (pl.col("assets") >= min_assets)
        & (pl.col("factor_unique") > 1) & (pl.col("return_unique") > 1),
    )
    retain(pl.col("date").is_in(usable["date"].implode()), "unusable_cross_section")
    counts.update(
        ic_rows=current.height, ic_days=usable.height,
        assets_min=usable["assets"].min(), assets_max=usable["assets"].max(),
    )
    return current, counts

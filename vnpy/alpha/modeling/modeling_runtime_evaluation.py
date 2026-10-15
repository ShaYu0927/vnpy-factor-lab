"""Evaluate configured formulas after the application's existing replay/import."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl

from vnpy.common.common_logger import get_logger
from vnpy.alpha.alpha_formula import Alpha
from vnpy.alpha.alpha_domain import FactorFrame, MarketFrame
from vnpy.alpha.expression import ExpressionAnalyzer
from vnpy.alpha.alpha_sample_diagnostics import eligible_samples, with_history_counts
from vnpy.alpha.alpha_time_split import (
    SessionSplit,
    TimeSplitConfig,
    build_session_split,
)
from vnpy.datafeed.datafeed_data_parquet_feed import ParquetDataFeed
from .modeling_alphalens_evaluator import AlphalensEvaluator
from .modeling_evaluation_config import FactorEvaluationConfig


def build_forward_labels(
    prices: pd.DataFrame, periods: tuple[int, ...], entry_offset: int
) -> pd.DataFrame:
    """Use the shared observed session calendar; never fill a missing stock price.

    At signal session t: buy close[t+entry_offset], sell close[t+entry_offset+p].
    Preserve both endpoint dates so train labels can be purged at the split.
    """
    if not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError("price dates and symbols must be unique")
    if entry_offset < 1:
        raise ValueError("close-based factors require entry_offset >= 1")
    prices = prices.sort_index().where(lambda x: np.isfinite(x) & (x > 0))
    index = pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=["date", "asset"]
    )
    labels = pd.DataFrame(index=index)
    entry = prices.shift(-entry_offset)
    dates = pd.Series(prices.index, index=prices.index)
    labels["entry_date"] = np.repeat(
        dates.shift(-entry_offset).to_numpy(), len(prices.columns)
    )
    for period in periods:
        values = prices.shift(-(entry_offset + period)) / entry - 1
        labels[f"{period}D"] = (
            values.replace([np.inf, -np.inf], np.nan).to_numpy().reshape(-1)
        )
        labels[f"exit_{period}D"] = np.repeat(
            dates.shift(-(entry_offset + period)).to_numpy(), len(prices.columns)
        )
    return labels


def _json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def evaluate_frames(
    market: pl.DataFrame,
    factors: pl.DataFrame,
    definitions: list[dict],
    config: FactorEvaluationConfig,
    *,
    context: dict | None = None,
    split: SessionSplit | None = None,
) -> dict:
    """Write per-factor, per-horizon, train/holdout diagnostics; do not select factors."""
    market_contract = MarketFrame(market)
    factor_contract = FactorFrame(factors)
    factor_contract.require_market_keys(market_contract)
    market, factors = market_contract.frame, factor_contract.frame
    started = perf_counter()
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=datetime.now(timezone.utc).strftime("evaluation_%Y%m%dT%H%M%SZ_"),
            dir=root,
        )
    )
    manifest = {
        "status": "running",
        "config": asdict(config),
        "context": context or {},
        "alphas": definitions,
        "metric": "Spearman Rank IC (Alphalens)",
        "evaluation_kind": "descriptive; no automatic factor selection or trading",
        "label": "close[t+entry_offset+period] / close[t+entry_offset] - 1",
        "calendar": "sorted observed market sessions; periods count sessions, not calendar days",
        "missing_prices": "no filling; nonpositive prices and zero-volume endpoints are excluded",
        "quantile_returns": "cross-section-demeaned cumulative holding-period returns; not annualized or fee-adjusted",
        "turnover": "Alphalens quantile membership turnover; lag counts available quantile dates",
        "rank_ic_ir": "daily mean / sample standard deviation; not annualized",
        "limitations": [
            "source close adjustment is not verified",
            "overlapping holding periods are not independent",
            "a short holdout is descriptive evidence, not proof of predictive ability",
        ],
        "versions": {
            name: version(name) for name in ("alphalens-reloaded", "pandas", "polars")
        },
    }
    _json(output / "manifest.json", manifest)
    try:
        if market.is_empty() or factors.is_empty() or not definitions:
            raise ValueError("evaluation requires market rows and factor definitions")
        market.write_parquet(output / "prices.parquet")
        factors.write_parquet(output / "factors.parquet")
        manifest["sha256"] = {}
        for name in ("prices.parquet", "factors.parquet"):
            with (output / name).open("rb") as stream:
                manifest["sha256"][name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        data = market.to_pandas()
        prices = AlphalensEvaluator.prepare_prices(data)
        if "volume" in data:
            volumes = AlphalensEvaluator.prepare_prices(data, price_column="volume")
            prices = prices.where(volumes > 0)
        labels = build_forward_labels(prices, config.periods, config.entry_offset)
        labels.reset_index().to_parquet(output / "labels.parquet", index=False)
        factor_frame = factors.to_pandas().set_index(["datetime", "vt_symbol"])
        factor_frame.index = factor_frame.index.set_names(["date", "asset"])
        if factor_frame.index.has_duplicates:
            raise ValueError("evaluation factors contain duplicate date/symbol rows")
        dates = prices.index
        split = split or build_session_split(
            dates.to_pydatetime(), TimeSplitConfig(config.train_fraction)
        )
        if tuple(dates.to_pydatetime()) != split.sessions:
            raise ValueError(
                "time split calendar does not match evaluation market data"
            )
        boundary = split.holdout_start
        history = (
            with_history_counts(market)
            .select("datetime", "vt_symbol", "history_bars")
            .to_pandas()
        )
        history = history.set_index(["datetime", "vt_symbol"])["history_bars"]
        history.index = history.index.set_names(["date", "asset"])
        if len(factor_frame.index.difference(history.index)):
            raise ValueError("evaluation factor keys must belong to market data")
        factor_frame = factor_frame.reindex(history.index)
        manifest.update(
            market_rows=market.height,
            symbols=len(prices.columns),
            sessions=len(dates),
            holdout_start=boundary.isoformat(),
            train_boundary="factor date and label exit must both precede holdout_start",
            time_split=split.as_dict(),
            sample_loss_order=[
                "missing_future",
                "boundary_dropped",
                "invalid_price",
                "warmup",
                "invalid_factor",
                "unusable_cross_section",
            ],
            label_capacity=[
                {
                    "segment": segment,
                    "period": period,
                    "sessions": len(split.dates_for(segment)),
                    "max_label_days": max(
                        0, len(split.dates_for(segment)) - config.entry_offset - period
                    ),
                }
                for segment in ("train", "holdout", "all")
                for period in config.periods
            ],
        )
        _json(output / "split.json", split.as_dict())
        get_logger("main.evaluation").info(
            "[evaluation/split] train=%d holdout=%d holdout_start=%s",
            split.train_count,
            len(split.dates_for("holdout")),
            boundary,
        )
        summaries, daily_rows, quantile_rows, turnover_rows = [], [], [], []
        for definition in definitions:
            name = definition["name"]
            factor = factor_frame[name].replace([np.inf, -np.inf], np.nan)
            lookback = (
                ExpressionAnalyzer()
                .analyze(Alpha(name, definition["formula"]).parse())
                .lookback
            )
            for period in config.periods:
                column = f"{period}D"
                joined = (
                    labels[[column, f"exit_{period}D"]].reindex(factor.index).copy()
                )
                joined["factor"] = factor
                joined["history_bars"] = history
                signal_dates = joined.index.get_level_values("date")
                for segment in ("all", "train", "holdout"):
                    mask = np.ones(len(joined), dtype=bool)
                    if segment == "train":
                        mask = signal_dates < boundary
                    elif segment == "holdout":
                        mask = signal_dates >= boundary
                    base = joined.loc[mask]
                    samples = pl.from_pandas(
                        base.reset_index().rename(
                            columns={
                                column: "label",
                                f"exit_{period}D": "exit_date",
                            }
                        )
                    )
                    usable, counts = eligible_samples(
                        samples,
                        input_days=len(split.dates_for(segment)),
                        lookback=lookback,
                        min_assets=config.min_assets,
                        train_end=boundary if segment == "train" else None,
                    )
                    valid = (
                        usable.to_pandas()
                        .set_index(["date", "asset"])
                        .rename(columns={"label": column})
                    )
                    assets_per_date = valid.groupby(level="date").size()
                    row = dict(
                        factor=name,
                        formula=definition.get("formula", ""),
                        period=period,
                        segment=segment,
                        **counts,
                        finite_factor_rows=int(base.factor.notna().sum()),
                        factor_coverage=_number(base.factor.notna().mean()),
                        boundary_dropped=counts["boundary_dropped_rows"],
                        rank_ic=None,
                        rank_ic_std=None,
                        rank_ic_ir=None,
                        positive_ic_fraction=None,
                        quantile_rows=0,
                        quantile_coverage=None,
                        top_minus_bottom=None,
                        quantile_status="unavailable",
                        status="skipped",
                        reason="",
                    )
                    if valid.empty:
                        row["reason"] = (
                            "no_valid_pairs"
                            if not counts["paired_rows"]
                            else "insufficient_or_constant_cross_sections"
                        )
                        summaries.append(row)
                        continue
                    evaluator = AlphalensEvaluator(
                        periods=(period,), quantiles=config.quantiles
                    )
                    report = evaluator.evaluate_forward_returns(
                        valid.factor, valid[[column]]
                    )
                    ic = report.information_coefficient[column].dropna()
                    std = ic.std()
                    row.update(
                        status="ok" if len(ic) else "skipped",
                        ic_days=len(ic),
                        rank_ic=_number(ic.mean()),
                        rank_ic_std=_number(std),
                        rank_ic_ir=_number(ic.mean() / std)
                        if pd.notna(std) and std > 0
                        else None,
                        positive_ic_fraction=_number((ic > 0).mean()),
                        quantile_rows=len(report.clean_data),
                        quantile_coverage=len(report.clean_data) / len(valid),
                        quantile_status="ok"
                        if len(report.clean_data) == len(valid)
                        else "partial"
                        if len(report.clean_data)
                        else "unavailable",
                    )
                    for date, value in ic.items():
                        daily_rows.append(
                            dict(
                                factor=name,
                                period=period,
                                segment=segment,
                                date=date.isoformat(),
                                rank_ic=float(value),
                                assets=int(assets_per_date.loc[date]),
                            )
                        )
                    if not report.clean_data.empty:
                        means = report.quantile_returns[column]
                        if 1 in means.index and config.quantiles in means.index:
                            row["top_minus_bottom"] = _number(
                                means.loc[config.quantiles] - means.loc[1]
                            )
                        for quantile, value in means.items():
                            quantile_rows.append(
                                dict(
                                    factor=name,
                                    period=period,
                                    segment=segment,
                                    quantile=int(quantile),
                                    mean_return=_number(value),
                                    standard_error=_number(
                                        report.quantile_standard_error.loc[
                                            quantile, column
                                        ]
                                    ),
                                )
                            )
                        for (quantile, lag), values in report.turnover.items():
                            for date, value in values.dropna().items():
                                turnover_rows.append(
                                    dict(
                                        factor=name,
                                        period=period,
                                        segment=segment,
                                        quantile=quantile,
                                        lag=lag,
                                        date=date.isoformat(),
                                        turnover=float(value),
                                    )
                                )
                    summaries.append(row)
            get_logger("main.evaluation").info(
                "[evaluation/factor] %s periods=%s complete", name, config.periods
            )
        pd.DataFrame(summaries).to_csv(
            output / "summary.csv", index=False, encoding="utf-8-sig"
        )
        for filename, rows, columns in (
            (
                "daily_rank_ic.csv",
                daily_rows,
                ["factor", "period", "segment", "date", "rank_ic", "assets"],
            ),
            (
                "quantile_returns.csv",
                quantile_rows,
                [
                    "factor",
                    "period",
                    "segment",
                    "quantile",
                    "mean_return",
                    "standard_error",
                ],
            ),
            (
                "turnover.csv",
                turnover_rows,
                ["factor", "period", "segment", "quantile", "lag", "date", "turnover"],
            ),
        ):
            pd.DataFrame(rows, columns=columns).to_csv(
                output / filename, index=False, encoding="utf-8-sig"
            )
        _json(output / "summary.json", summaries)
        manifest.update(
            status="complete",
            elapsed_seconds=round(perf_counter() - started, 3),
            result_rows=len(summaries),
            evaluated_rows=sum(r["status"] == "ok" for r in summaries),
            skipped_rows=sum(r["status"] == "skipped" for r in summaries),
        )
        from .modeling_evaluation_report import write_report

        write_report(output / "report.html", manifest, summaries, daily_rows)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _json(output / "manifest.json", manifest)
    result = {
        "output": str(output),
        "report": str(output / "report.html"),
        "evaluated_rows": manifest["evaluated_rows"],
        "skipped_rows": manifest["skipped_rows"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
    get_logger("main.evaluation").info("[evaluation/complete] %s", result)
    return result


def run_factor_evaluation(runtime, options, engine, definitions, *, snapshot=None):
    """Batch-recompute each formula independently of strategy sample filtering."""
    if snapshot is not None:
        paths = sorted(Path(snapshot["root"]).glob("????-??-??.parquet"))
        market = pl.concat(
            [pl.read_parquet(path) for path in paths], how="diagonal_relaxed"
        )
    else:
        setting = runtime.parquet
        feed = ParquetDataFeed(setting.root)
        frames = []
        # Convert bounded bar batches; evaluation materializes the configured
        # research period as a columnar table, not the strategy's capped cache.
        for batch in feed.iter_batches(
            batch_size=10000,
            start=setting.start,
            end=setting.end,
            symbols=setting.symbols,
            markets=setting.markets,
            frequency=setting.frequency,
            skip_zero_volume=setting.skip_zero_volume,
            skip_invalid_ohlc=setting.skip_invalid_ohlc,
            allow_missing_years=setting.allow_missing_years,
        ):
            grouped = {}
            for bar in batch:
                grouped.setdefault(bar.symbol, []).append(bar)
            frames.append(engine.from_bars(grouped))
        if not frames:
            raise ValueError("no market data available for factor evaluation")
        market = pl.concat(frames, how="diagonal_relaxed")
    factors = engine.calculate(market)
    split = build_session_split(market["datetime"].to_list(), runtime.time_split)
    options = replace(options, train_fraction=runtime.time_split.train_fraction)
    return evaluate_frames(
        market,
        factors,
        definitions,
        options,
        split=split,
        context={
            "runtime": runtime.raw,
            "input": asdict(runtime.parquet),
            "factor_source": "same configured formulas, independently batch-calculated before strategy completeness filtering",
        },
    )

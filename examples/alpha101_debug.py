"""Run from the project directory: python -m examples.alpha101_debug.

Use --csv daily.csv for real daily data with datetime,vt_symbol,OHLCV columns.
Use --factors 1 2 101 to calculate a subset; defaults to three factors.
"""
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from vnpy.alpha import AlphaEngine
from vnpy.alpha.modeling import AlphaAnalyzer, AlphaDatasetBuilder
from vnpy.common.logger import init_global_logger, get_logger, shutdown_global_logger


def demo_frame() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for symbol, base in (("A", 10), ("B", 20), ("C", 30)):
        close = float(base)
        for day in range(320):
            opening = close * (1 + rng.normal(0, 0.01))
            close = opening * (1 + rng.normal(0, 0.02))
            high = max(opening, close) * 1.01
            low = min(opening, close) * 0.99
            rows.append({
                "datetime": datetime(2025, 1, 1) + timedelta(days=day),
                "vt_symbol": symbol, "open": opening, "high": high,
                "low": low, "close": close, "volume": float(rng.integers(1000, 10000)),
                "vwap": (opening + close) / 2, "market_cap": close * 1e8,
                "industry": "DEMO", "sector": "DEMO", "subindustry": "DEMO",
            })
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--entry-offset", type=int, default=1)
    parser.add_argument("--min-assets", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    init_global_logger(level=logging.DEBUG if args.debug else logging.INFO)
    logger = get_logger("alpha.demo")
    try:
        frame = pl.read_csv(args.csv, try_parse_dates=True) if args.csv else demo_frame()
        logger.info("Data source: %s", args.csv or "synthetic demonstration data")
        engine = AlphaEngine([])
        features = engine.calculate_alpha101(frame, args.factors)
        feature_names = [name for name in features.columns if name.startswith("alpha")]
        dataset = AlphaDatasetBuilder(
            horizon=args.horizon,
            entry_offset=args.entry_offset,
        ).build(features, frame, feature_names=feature_names)
        report = AlphaAnalyzer(min_assets=args.min_assets).evaluate(
            dataset,
            feature_names,
        )
        logger.info("Alpha features: rows=%d factors=%d", features.height, len(feature_names))
        logger.info("Labeled dataset: rows=%d", dataset.height)
        for metric in report.metrics:
            logger.info(
                "%s IC=%.6f ICIR=%.6f RankIC=%.6f RankICIR=%.6f dates=%d",
                metric.factor_name,
                metric.mean_ic,
                metric.ic_ir,
                metric.mean_rank_ic,
                metric.rank_ic_ir,
                metric.observations,
            )
    finally:
        shutdown_global_logger()


if __name__ == "__main__":
    main()

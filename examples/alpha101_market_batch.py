"""python -m examples.alpha101_market_batch --input daily.parquet --date 2025-11-16

With no arguments, run synthetic daily data through the actual event engine.
"""
import argparse
from pathlib import Path
from threading import Event

import polars as pl

from examples.alpha101_debug import demo_frame
from vnpy.common.logger import init_global_logger, get_logger, shutdown_global_logger
from vnpy.datafeed.daily_store import DailyMarketStore
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.batch_module import daily_alpha_batch_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--date", default="2025-11-16")
    parser.add_argument("--root", type=Path, default=Path("data/alpha101_demo"))
    parser.add_argument("--universe", type=Path, help="Text file: one expected active symbol per line")
    args = parser.parse_args()
    if args.input and not args.universe:
        parser.error("real data requires --universe to detect missing symbols")
    init_global_logger()
    logger = get_logger("batch.demo")
    engine = ModuleEngine()
    done = Event()
    outcome = []

    def consume(ctx, event):
        if event.event_type == EventType.FACTOR_BATCH_FAILED:
            logger.error("[flow/failed] request=%s error=%s", event.get("request_id"), event.get("error"))
            outcome.append(event.get("error"))
            done.set()
        elif event.event_type == EventType.FACTOR_BATCH_READY:
            logger.info("[flow/consume] request=%s received; reading result file", event.get("request_id"))
            result = pl.read_parquet(event.get("result_path"))
            get_logger("batch.demo").info("Batch received: shape=%s path=%s", result.shape, event.get("result_path"))
            preview = [name for name in ("datetime", "vt_symbol", "close", "alpha001", "alpha101") if name in result.columns]
            logger.info("[flow/preview] first_3_rows=%s", result.select(preview).head(3).to_dicts())
            done.set()

    try:
        logger.info("[flow/start] source=%s date=%s root=%s", args.input or "synthetic demo data", args.date, args.root.resolve())
        store = DailyMarketStore(args.root / "daily")
        if args.input:
            store.import_file(args.input)
            symbols = [line.strip() for line in args.universe.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            store.upsert(demo_frame())
            symbols = ["A", "B", "C"]
        engine.register_module("daily_alpha", daily_alpha_batch_entry, queue_size=8, config={
            "daily_store": str(store.root), "result_store": str(args.root / "factors"),
            "history_dates": 320, "batch_targets": ["batch_consumer"],
        })
        engine.register_module("batch_consumer", consume, queue_size=8)
        engine.start_all()
        logger.info("[flow/modules] started daily_alpha -> batch_consumer")
        request = EngineEvent(EventType.MARKET_DATA_READY, {
            "trade_date": args.date, "expected_symbols": symbols, "excluded_symbols": {},
        })
        logger.info("[flow/submit] request=%s event=MARKET_DATA_READY expected_symbols=%d", request.event_id, len(symbols))
        if not engine.post_event("daily_alpha", request):
            raise RuntimeError("batch request queue rejected request")
        while not done.wait(10):
            get_logger("batch.demo").info("Waiting for daily factor batch...")
        if outcome:
            raise RuntimeError(outcome[0])
    finally:
        engine.stop_all()
        logger.info("[flow/stop] modules stopped")
        shutdown_global_logger()


if __name__ == "__main__":
    main()

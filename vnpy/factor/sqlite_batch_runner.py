"""Run the configured SQLite source through daily batch events."""
from pathlib import Path
from threading import Event
from uuid import uuid4

import polars as pl

from vnpy.common.logger import get_logger
from vnpy.datafeed.daily_store import DailyMarketStore
from vnpy.datafeed.gm_sqlite_datafeed import GmSqliteDataFeed
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.batch_module import daily_alpha_batch_entry


def run_sqlite_alpha101(config, options, *, feed=None):
    logger = get_logger("main.alpha101")
    expected = [item.strip() for item in (config.symbols or "").split(",") if item.strip()]
    if not expected:
        raise ValueError("Alpha101 requires gm_sqlite.symbols as the expected universe")
    history_dates = int(options.get("history_dates", 320))
    if history_dates < 1:
        raise ValueError("history_dates must be positive")
    project = Path(__file__).resolve().parents[2]
    root = Path(options.get("root", "data/main_alpha101"))
    if not root.is_absolute():
        root = project / root
    # Each import is a separate snapshot; old rows cannot hide missing source data.
    store = DailyMarketStore(root / "imports" / uuid4().hex)
    source = feed or GmSqliteDataFeed(config.root)
    logger.info("[main/alpha101] source=%s symbols=%d latest-date-only=True history_dates=%d",
                config.root, len(expected), history_dates)
    rows = []
    count = 0
    latest = None
    for bar in source.iter_history(
        start=config.start, end=config.end, symbols=expected, markets=config.markets,
        frequency=config.frequency, skip_zero_volume=config.skip_zero_volume,
        skip_invalid_ohlc=config.skip_invalid_ohlc, allow_missing_years=config.allow_missing_years,
    ):
        day = bar.bob.date()
        latest = day if latest is None else max(latest, day)
        row = {"datetime": day, "vt_symbol": bar.symbol,
               **{field: getattr(bar, field) for field in ("open", "high", "low", "close", "volume")}}
        for field in ("vwap", "market_cap", "industry", "sector", "subindustry"):
            if field in bar.extra:
                row[field] = bar.extra[field]
        rows.append(row)
        count += 1
        if count == 1 or count % max(1, config.progress_every) == 0:
            logger.info("[main/import] rows=%d latest=%s", count, latest)
        if len(rows) >= 100_000:
            store.upsert(pl.DataFrame(rows))
            rows.clear()
    if rows:
        store.upsert(pl.DataFrame(rows))
    if latest is None:
        raise ValueError("SQLite source returned no matching daily bars")
    logger.info("[main/import] complete rows=%d calculation_date=%s snapshot=%s", count, latest, store.root)
    engine = ModuleEngine()
    completed = Event()
    outcome = {}

    def consume(ctx, event):
        if event.event_type == EventType.FACTOR_BATCH_FAILED:
            outcome["error"] = event.get("error")
            completed.set()
        elif event.event_type == EventType.FACTOR_BATCH_READY:
            try:
                outcome["result"] = dict(event.data)
                result = pl.read_parquet(event.get("result_path"))
                names = [name for name in ("vt_symbol", "close", "alpha001", "alpha101") if name in result.columns]
                logger.info("[main/result] date=%s shape=%s path=%s", event.get("trade_date"), result.shape, event.get("result_path"))
                logger.info("[main/preview] %s", result.select(names).head(5).to_dicts())
            except Exception as exc:
                outcome["error"] = str(exc)
            finally:
                completed.set()

    engine.register_module("daily_alpha", daily_alpha_batch_entry, queue_size=8, config={
        "daily_store": str(store.root), "result_store": str(root / "factors"),
        "history_dates": history_dates, "alpha101_factors": options.get("factors"),
        "batch_targets": ["alpha101_result"],
    })
    engine.register_module("alpha101_result", consume, queue_size=8)
    engine.start_all()
    try:
        request = EngineEvent(EventType.MARKET_DATA_READY, {
            "trade_date": latest.isoformat(), "expected_symbols": expected,
            "excluded_symbols": options.get("excluded_symbols", {}),
        }, source="sqlite_import")
        logger.info("[main/submit] request=%s event=MARKET_DATA_READY date=%s", request.event_id, latest)
        if not engine.post_event("daily_alpha", request):
            raise RuntimeError("daily_alpha queue rejected request")
        while not completed.wait(10):
            logger.info("[main/wait] waiting for Alpha101 result request=%s", request.event_id)
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        return outcome["result"]
    finally:
        engine.stop_all()

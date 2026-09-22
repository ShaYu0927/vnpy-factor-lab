"""Import Parquet history into daily local snapshots and evaluate configured alphas."""

from pathlib import Path
from uuid import uuid4

import polars as pl

from vnpy.common.logger import get_logger
from vnpy.datafeed.daily_store import DailyMarketStore, atomic_parquet
from vnpy.datafeed.parquet_datafeed import ParquetDataFeed


def _resolve_output_root(options):
    project = Path(__file__).resolve().parents[2]
    root = Path(options.get("root", "data/local_market"))
    if not root.is_absolute():
        root = project / root
    return root


def import_parquet_history(config, options, *, feed=None):
    """Read Parquet history and save daily Parquet files; return snapshot metadata."""
    logger = get_logger("main.parquet_import")
    symbols = [
        item.strip()
        for item in (config.symbols or "").split(",")
        if item.strip()
    ]
    root = _resolve_output_root(options)
    # Each import is a separate snapshot; old rows cannot hide missing source data.
    store = DailyMarketStore(root / "imports" / uuid4().hex)
    source = feed or ParquetDataFeed(config.root)

    rows = []
    count = 0
    latest = None
    for bar in source.iter_history(
        start=config.start,
        end=config.end,
        symbols=symbols,
        markets=config.markets,
        frequency=config.frequency,
        skip_zero_volume=config.skip_zero_volume,
        skip_invalid_ohlc=config.skip_invalid_ohlc,
        allow_missing_years=config.allow_missing_years,
    ):
        day = bar.bob.date()
        latest = day if latest is None else max(latest, day)
        row = {
            "datetime": day,
            "vt_symbol": bar.symbol,
            **{
                field: getattr(bar, field)
                for field in ("open", "high", "low", "close", "volume")
            },
        }
        row["amount"] = bar.amount
        for field in ("pre_close", "position", "updated_at", "ext_data", "source_file"):
            row[field] = bar.extra.get(field)
        for field in ("vwap", "market_cap", "industry", "sector", "subindustry"):
            if field in bar.extra:
                row[field] = bar.extra[field]
        rows.append(row)
        count += 1
        if len(rows) >= 100_000:
            store.upsert(pl.DataFrame(rows))
            rows.clear()

    if rows:
        store.upsert(pl.DataFrame(rows))
    if latest is None:
        raise ValueError("Parquet source returned no matching daily bars")

    logger.info(
        "[main/import] rows=%d latest=%s root=%s",
        count,
        latest,
        store.root,
    )
    return {
        "rows": count,
        "latest_date": latest.isoformat(),
        "root": str(store.root),
    }


def calculate_imported_alphas(snapshot, engine) -> Path:
    """Evaluate a complete import snapshot, including cross-date rolling history."""
    store = DailyMarketStore(snapshot["root"])
    history_dates = len(list(store.root.glob("????-??-??.parquet")))
    frame = store.load_window(snapshot["latest_date"], history_dates=history_dates)
    result = engine.calculate(frame)
    output = store.root / "factors.parquet"
    atomic_parquet(result, output)
    get_logger("main.parquet_import").info(
        "[main/alphas] rows=%d factors=%d output=%s",
        result.height, len(engine.definitions), output,
    )
    return output

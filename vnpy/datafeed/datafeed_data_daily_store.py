"""Single-writer daily Parquet store for bulk history and daily upserts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import polars as pl

def atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DailyMarketStore:
    """
    One partition per date. Upsert replaces complete rows by symbol/date.

    Input datetime denotes a trading DATE, not an intraday timestamp. Use one
    writer, and finish imports before posting MARKET_DATA_READY for that date.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def upsert(self, frame: pl.DataFrame) -> int:
        required = {"datetime", "vt_symbol", "open", "high", "low", "close", "volume"}
        if missing := required - set(frame.columns):
            raise ValueError(f"daily input missing columns: {sorted(missing)}")
        if frame.is_empty():
            return 0
        dt = pl.col("datetime")
        if frame.schema["datetime"] == pl.String:
            dt = dt.str.to_datetime(strict=True)
        frame = frame.with_columns(dt.cast(pl.Date), pl.col("vt_symbol").cast(pl.String))
        if frame["datetime"].null_count() or frame["vt_symbol"].null_count():
            raise ValueError("daily keys must not be null")
        if frame.filter(pl.col("vt_symbol").str.strip_chars() == "").height:
            raise ValueError("daily symbol must not be empty")
        for partition in frame.partition_by("datetime"):
            day = partition["datetime"][0]
            path = self.root / f"{day.isoformat()}.parquet"
            if path.exists():
                partition = pl.concat([pl.read_parquet(path), partition], how="diagonal_relaxed")
            partition = partition.unique(["datetime", "vt_symbol"], keep="last").sort("vt_symbol")
            atomic_parquet(partition, path)
        return frame.height

    def import_file(self, path: str | Path, batch_size: int = 100_000) -> int:
        """Stream CSV/Parquet row batches; never enqueue individual historical bars."""
        path = Path(path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            with pq.ParquetFile(path) as file:
                return sum(self.upsert(pl.from_arrow(batch)) for batch in file.iter_batches(batch_size=batch_size))
        if path.suffix.lower() == ".csv":
            import pandas as pd

            with pd.read_csv(path, chunksize=batch_size) as batches:
                return sum(self.upsert(pl.from_pandas(batch)) for batch in batches)
        raise ValueError("daily import supports .csv and .parquet")

    def load_window(self, trade_date: str, history_dates: int = 320) -> pl.DataFrame:
        day = date.fromisoformat(trade_date)
        if history_dates < 1:
            raise ValueError("history_dates must be positive")
        files = sorted(path for path in self.root.glob("????-??-??.parquet") if path.stem <= day.isoformat())
        if not files or files[-1].stem != day.isoformat():
            raise ValueError(f"no daily partition for {day}")
        # Keep historical membership per date; do not filter history to today's universe.
        selected = files[-history_dates:]
        return pl.concat([pl.read_parquet(path) for path in selected], how="diagonal_relaxed")

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

from vnpy.config.runtime_config import GmSqliteBatchConfig
from vnpy.datafeed.gm_sqlite_datafeed import GmSqliteDataFeed
from vnpy.datafeed.model import BarData
from vnpy.factor.core.factor_engine import (
    ExecutionMode,
    Factor,
    FactorContext,
    FactorEngine,
    FactorValue,
)
from vnpy.factor.realtime_service import BasicFactorSet


@dataclass(slots=True)
class BatchFactorSummary:
    bars_read: int = 0
    dates_processed: int = 0
    symbols_seen: int = 0
    values_written: int = 0
    factor_errors: int = 0


class CsvFactorResultWriter:
    """Stream normalized factor values to a stable long-form CSV file."""

    fieldnames = (
        "trade_date",
        "symbol",
        "factor_name",
        "value",
        "primary_field",
        "status",
        "reason",
        "version",
        "fields_json",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        append: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if append and overwrite:
            raise ValueError("append and overwrite cannot both be enabled")
        if self.path.exists() and not overwrite:
            if not append:
                raise FileExistsError(
                    f"batch factor output already exists: {self.path}; "
                    "set overwrite=true to replace it or resume=true to append"
                )
            self._validate_header()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if append else "w"
        self._file: TextIO = self.path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not file_exists or not append:
            self._writer.writeheader()

    def _validate_header(self) -> None:
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), None)
        if tuple(header or ()) != self.fieldnames:
            raise ValueError(
                f"batch factor output has an incompatible header: {self.path}"
            )

    def write(self, values: Iterable[FactorValue]) -> int:
        count = 0
        for value in values:
            self._writer.writerow(
                {
                    "trade_date": value.trade_date or "",
                    "symbol": value.symbol,
                    "factor_name": value.factor_name,
                    "value": "" if value.value is None else value.value,
                    "primary_field": value.primary_field or "",
                    "status": value.status.value,
                    "reason": value.reason,
                    "version": value.version,
                    "fields_json": json.dumps(
                        value.fields,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
            count += 1
        return count

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CsvFactorResultWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class SqliteFactorResultWriter:
    """Persist normalized factor values with idempotent batch inserts."""

    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() and overwrite:
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_values (
                trade_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                value REAL,
                primary_field TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                version TEXT,
                fields_json TEXT NOT NULL,
                PRIMARY KEY (trade_date, symbol, factor_name)
            ) WITHOUT ROWID
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_factor_symbol_date "
            "ON factor_values(symbol, trade_date)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_factor_name_date "
            "ON factor_values(factor_name, trade_date)"
        )
        self._connection.commit()

    def write(self, values: Iterable[FactorValue]) -> int:
        rows = [
            (
                value.trade_date or "",
                value.symbol,
                value.factor_name,
                value.value,
                value.primary_field or "",
                value.status.value,
                value.reason,
                value.version,
                json.dumps(
                    value.fields,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            for value in values
        ]
        self._connection.executemany(
            """
            INSERT INTO factor_values (
                trade_date, symbol, factor_name, value, primary_field,
                status, reason, version, fields_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_date, symbol, factor_name) DO UPDATE SET
                value=excluded.value,
                primary_field=excluded.primary_field,
                status=excluded.status,
                reason=excluded.reason,
                version=excluded.version,
                fields_json=excluded.fields_json
            """,
            rows,
        )
        self._connection.commit()
        return len(rows)

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> SqliteFactorResultWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class ConsoleFactorResultWriter:
    """Keep only the latest snapshot in memory and print it on completion."""

    def __init__(self) -> None:
        self._latest_date: str = ""
        self._latest_values: list[FactorValue] = []

    def write(self, values: Iterable[FactorValue]) -> int:
        rows = list(values)
        if not rows:
            return 0

        latest_date = max(value.trade_date or "" for value in rows)
        if latest_date != self._latest_date:
            self._latest_date = latest_date
            self._latest_values = []
        self._latest_values.extend(
            value for value in rows if (value.trade_date or "") == latest_date
        )
        return len(rows)

    def close(self) -> None:
        print(f"[factor results] latest_date={self._latest_date}", flush=True)
        print(
            f"{'symbol':<14} {'factor':<16} {'value':>12} {'status':<12}",
            flush=True,
        )
        for value in sorted(
            self._latest_values,
            key=lambda item: (item.symbol, item.factor_name),
        ):
            display_value = "None" if value.value is None else f"{value.value:.6f}"
            print(
                f"{value.symbol:<14} {value.factor_name:<16} "
                f"{display_value:>12} {value.status.value:<12}",
                flush=True,
            )

    def __enter__(self) -> ConsoleFactorResultWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def iter_bars_by_date(
    batches: Iterable[Sequence[BarData]],
) -> Iterator[tuple[date, list[BarData]]]:
    """Preserve a date group even when an input batch splits it."""
    current_date: date | None = None
    current_bars: list[BarData] = []

    for batch in batches:
        for bar in batch:
            bar_date = bar.bob.date()
            if current_date is not None and bar_date != current_date:
                yield current_date, current_bars
                current_bars = []
            current_date = bar_date
            current_bars.append(bar)

    if current_date is not None:
        yield current_date, current_bars


def select_factors(names: Sequence[str]) -> tuple[Factor, ...]:
    available = {factor.name: factor for factor in BasicFactorSet.create()}
    unknown = sorted(set(names) - available.keys())
    if unknown:
        supported = ", ".join(sorted(available))
        raise ValueError(
            f"unknown batch factors: {', '.join(unknown)}; supported: {supported}"
        )
    return tuple(available[name] for name in names)


def read_last_trade_date(path: str | Path) -> date | None:
    """Read the last persisted trade date without scanning a huge result."""
    output = Path(path).expanduser().resolve()
    if not output.is_file() or output.stat().st_size == 0:
        return None
    if output.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        connection = sqlite3.connect(f"file:{output.as_posix()}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT MAX(trade_date) FROM factor_values"
            ).fetchone()
        finally:
            connection.close()
        return date.fromisoformat(row[0]) if row and row[0] else None

    with output.open("rb") as file:
        position = file.seek(0, 2)
        buffer = b""
        while position > 0 and buffer.count(b"\n") < 2:
            size = min(8192, position)
            position -= size
            file.seek(position)
            buffer = file.read(size) + buffer

    lines = [line for line in buffer.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    row = next(csv.reader([lines[-1].decode("utf-8")]))
    try:
        return date.fromisoformat(row[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid final CSV record in {output}") from exc


def create_result_writer(
    path: str | Path,
    *,
    overwrite: bool,
    resume: bool,
) -> CsvFactorResultWriter | SqliteFactorResultWriter | ConsoleFactorResultWriter:
    if str(path).strip().lower() in {"console", ":console:"}:
        return ConsoleFactorResultWriter()
    output = Path(path)
    if output.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return SqliteFactorResultWriter(output, overwrite=overwrite and not resume)
    return CsvFactorResultWriter(
        output,
        overwrite=overwrite and not resume,
        append=resume and output.is_file(),
    )


def run_gm_sqlite_batch(config: GmSqliteBatchConfig) -> BatchFactorSummary:
    """Calculate date snapshots from local GM SQLite files in bounded memory."""
    factors = select_factors(config.factors)
    required_bars = max(factor.min_bars for factor in factors)
    if config.window_size < required_bars:
        raise ValueError(
            f"window_size={config.window_size} is too small for selected factors; "
            f"at least {required_bars} bars are required"
        )

    feed = GmSqliteDataFeed(config.root)
    engine = FactorEngine(
        factors=factors,
        mode=ExecutionMode(config.factor_mode),
        max_workers=config.max_workers,
    )
    windows: dict[str, deque[BarData]] = defaultdict(
        lambda: deque(maxlen=config.window_size)
    )
    summary = BatchFactorSummary()
    resume_date = read_last_trade_date(config.output) if config.resume else None
    if resume_date is not None:
        configured_end = GmSqliteDataFeed._parse_datetime(config.end).date()
        if resume_date >= configured_end:
            print(
                f"[GM sqlite batch] already completed through {resume_date}; "
                "nothing to resume",
                flush=True,
            )
            return summary
        print(
            f"[GM sqlite batch] resuming after {resume_date}; "
            "rebuilding rolling windows from source data",
            flush=True,
        )

    batches = feed.iter_batches(
        batch_size=config.batch_size,
        start=config.start,
        end=config.end,
        symbols=config.symbols,
        markets=config.markets,
        frequency=config.frequency,
        skip_zero_volume=config.skip_zero_volume,
        skip_invalid_ohlc=config.skip_invalid_ohlc,
        allow_missing_years=config.allow_missing_years,
    )

    try:
        with create_result_writer(
            config.output,
            overwrite=config.overwrite,
            resume=config.resume,
        ) as writer:
            for trade_date, day_bars in iter_bars_by_date(batches):
                updated_symbols: set[str] = set()
                for bar in day_bars:
                    windows[bar.symbol].append(bar)
                    updated_symbols.add(bar.symbol)

                summary.bars_read += len(day_bars)
                summary.dates_processed += 1
                summary.symbols_seen = len(windows)

                if resume_date is not None and trade_date <= resume_date:
                    continue

                symbol_data_map = {
                    symbol: list(windows[symbol])
                    for symbol in updated_symbols
                    if len(windows[symbol]) >= required_bars
                }
                if symbol_data_map:
                    result = engine.calculate_many(
                        symbol_data_map,
                        context=FactorContext(trade_date=trade_date.isoformat()),
                    )
                    summary.values_written += writer.write(result.values)
                    summary.factor_errors += len(result.errors)
                    if result.errors:
                        first = result.errors[0]
                        raise RuntimeError(
                            f"batch factor calculation failed for {first.symbol}/"
                            f"{first.factor_name}: {first.error}"
                        )

                if (
                    summary.dates_processed % config.progress_every_dates == 0
                ):
                    print(
                        "[GM sqlite batch] "
                        f"dates={summary.dates_processed} "
                        f"bars={summary.bars_read} "
                        f"symbols={summary.symbols_seen} "
                        f"values={summary.values_written} "
                        f"last_date={trade_date}",
                        flush=True,
                    )
    finally:
        executor = getattr(engine, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

    output_display = (
        "console"
        if str(config.output).strip().lower() in {"console", ":console:"}
        else str(Path(config.output).expanduser().resolve())
    )
    print(
        "[GM sqlite batch] completed "
        f"dates={summary.dates_processed} "
        f"bars={summary.bars_read} "
        f"symbols={summary.symbols_seen} "
        f"values={summary.values_written} "
        f"output={output_display}",
        flush=True,
    )
    return summary

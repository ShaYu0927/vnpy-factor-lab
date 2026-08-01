from __future__ import annotations

import heapq
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from vnpy.datafeed.model import BarData, BarSource, MarketBar


DAY_BAR_FILE_PATTERN = re.compile(r"^(?P<market>[A-Z]+)_(?P<year>\d{4})\.dat$")
CHINA_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_STOCK_MARKETS = ("SHSE", "SZSE")


class GmSqliteDataFeed:
    """Stream GM daily bars directly from downloaded yearly SQLite files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.day_bar_dir = self._resolve_day_bar_dir(self.root)

    def iter_history(
        self,
        *,
        start: str | datetime,
        end: str | datetime,
        symbols: str | Iterable[str] | None = None,
        markets: str | Iterable[str] | None = None,
        frequency: str = "1d",
        skip_zero_volume: bool = False,
        skip_invalid_ohlc: bool = False,
        allow_missing_years: bool = False,
    ) -> Iterator[BarData]:
        """Yield globally date-sorted bars without materializing the full result."""
        if frequency != "1d":
            raise ValueError("GM SQLite day_bar files only support frequency='1d'")

        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        if start_dt > end_dt:
            raise ValueError("start must not be later than end")

        symbol_list = self._normalize_values(symbols)
        market_list = self._resolve_markets(symbol_list, markets)
        files = self.resolve_files(
            start=start_dt,
            end=end_dt,
            markets=market_list,
            allow_missing_years=allow_missing_years,
        )

        streams: list[Iterator[BarData]] = []
        for path in files:
            match = DAY_BAR_FILE_PATTERN.match(path.name)
            assert match is not None
            market = match.group("market")
            file_symbols = [
                symbol for symbol in symbol_list if symbol.partition(".")[0] == market
            ]
            if symbol_list and not file_symbols:
                continue

            streams.append(
                self._iter_file(
                    path=path,
                    start_date=start_dt.date().isoformat(),
                    end_date=end_dt.date().isoformat(),
                    symbols=file_symbols,
                    frequency=frequency,
                    skip_zero_volume=skip_zero_volume,
                    skip_invalid_ohlc=skip_invalid_ohlc,
                )
            )

        yield from heapq.merge(
            *streams,
            key=lambda bar: (bar.bob, bar.symbol),
        )

    def iter_batches(
        self,
        *,
        batch_size: int = 10_000,
        **history_kwargs: Any,
    ) -> Iterator[list[BarData]]:
        """Yield bounded lists for vectorized or persistence-oriented consumers."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        batch: list[BarData] = []
        for bar in self.iter_history(**history_kwargs):
            batch.append(bar)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def load_history(self, **history_kwargs: Any) -> list[BarData]:
        """Compatibility helper for small queries; prefer iter_history at scale."""
        return list(self.iter_history(**history_kwargs))

    def resolve_files(
        self,
        *,
        start: str | datetime,
        end: str | datetime,
        markets: str | Iterable[str] | None = None,
        allow_missing_years: bool = False,
    ) -> list[Path]:
        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        market_list = self._normalize_values(markets) or list(DEFAULT_STOCK_MARKETS)
        requested = {
            (market.upper(), year)
            for market in market_list
            for year in range(start_dt.year, end_dt.year + 1)
        }
        available = self._available_files()
        missing = sorted(requested - available.keys())

        if missing and not allow_missing_years:
            display = ", ".join(f"{market}_{year}" for market, year in missing)
            raise FileNotFoundError(
                f"GM day-bar years are missing under {self.day_bar_dir}: {display}"
            )

        return [available[key] for key in sorted(requested) if key in available]

    def _iter_file(
        self,
        *,
        path: Path,
        start_date: str,
        end_date: str,
        symbols: list[str],
        frequency: str,
        skip_zero_volume: bool,
        skip_invalid_ohlc: bool,
    ) -> Iterator[BarData]:
        sql = (
            "SELECT symbol, trade_date, sec_id, open, high, low, close, volume, "
            "amount, pre_close, Position, updated_at, ext_data "
            "FROM dists_day_bar WHERE trade_date >= ? AND trade_date <= ? "
            "AND close IS NOT NULL"
        )
        params: list[object] = [start_date, end_date]

        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            sql += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        if skip_zero_volume:
            sql += " AND volume > 0"
        if skip_invalid_ohlc:
            sql += (
                " AND high >= low AND open BETWEEN low AND high "
                "AND close BETWEEN low AND high"
            )
        sql += " ORDER BY trade_date ASC, symbol ASC"

        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            for row in connection.execute(sql, params):
                bob = datetime.fromisoformat(row["trade_date"]).replace(tzinfo=CHINA_TZ)
                yield MarketBar(
                    symbol=row["symbol"],
                    bob=bob,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    amount=row["amount"],
                    frequency=frequency,
                    source=BarSource.GM_SQLITE.value,
                    extra={
                        "sec_id": row["sec_id"],
                        "pre_close": row["pre_close"],
                        "position": row["Position"],
                        "updated_at": row["updated_at"],
                        "ext_data": row["ext_data"],
                        "source_file": str(path),
                    },
                )
        finally:
            connection.close()

    def _available_files(self) -> dict[tuple[str, int], Path]:
        files: dict[tuple[str, int], Path] = {}
        for path in self.day_bar_dir.glob("*.dat"):
            match = DAY_BAR_FILE_PATTERN.match(path.name)
            if match:
                files[(match.group("market"), int(match.group("year")))] = path
        return files

    @staticmethod
    def _resolve_day_bar_dir(root: Path) -> Path:
        candidates = (
            root,
            root / "day_bar",
            root / "basic_data" / "day_bar",
        )
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*_????.dat")):
                return candidate
        raise FileNotFoundError(f"GM day_bar directory was not found under: {root}")

    @staticmethod
    def _normalize_values(values: str | Iterable[str] | None) -> list[str]:
        if values is None:
            return []
        candidates = values.split(",") if isinstance(values, str) else values
        return [str(value).strip().upper() for value in candidates if str(value).strip()]

    @classmethod
    def _resolve_markets(
        cls,
        symbols: list[str],
        markets: str | Iterable[str] | None,
    ) -> list[str]:
        configured = cls._normalize_values(markets)
        symbol_markets = sorted({symbol.partition(".")[0] for symbol in symbols})
        if configured and symbol_markets:
            invalid = sorted(set(symbol_markets) - set(configured))
            if invalid:
                raise ValueError(
                    f"symbol markets are not enabled by markets: {', '.join(invalid)}"
                )
        return symbol_markets or configured or list(DEFAULT_STOCK_MARKETS)

    @staticmethod
    def _parse_datetime(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

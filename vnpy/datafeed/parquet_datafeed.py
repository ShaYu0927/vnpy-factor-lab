from __future__ import annotations

import heapq
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from datetime import datetime
from itertools import groupby
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.common.logger import get_logger
from vnpy.datafeed.model import BarData, BarSource, MarketBar


DAY_BAR_DIRECTORY_PATTERN = re.compile(r"^(?P<market>[A-Z]+)_(?P<year>\d{4})$")
CHINA_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_STOCK_MARKETS = ("SHSE", "SZSE")


class ParquetDataFeed:
    """Stream GM daily bars directly from local yearly Parquet files."""

    def __init__(self, root: str | Path) -> None:
        self.logger = get_logger("datafeed.parquet")
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
        """Yield date-sorted bars, materializing filtered data one year at a time."""
        if frequency != "1d":
            raise ValueError("Parquet day_bar files only support frequency='1d'")

        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        if start_dt > end_dt:
            raise ValueError("start must not be later than end")

        symbol_list = self._normalize_values(symbols)
        market_list = self._resolve_markets(symbol_list, markets)
        files = self.resolve_files(
            start=start_dt, end=end_dt, markets=market_list,
            allow_missing_years=allow_missing_years,
        )
        # Read one year at a time so a multi-decade query does not retain every
        # year's sorted data in memory while merging markets.
        count = 0
        for year, year_files in groupby(files, key=lambda path: path.parent.name[-4:]):
            streams: list[Generator[BarData, None, None]] = []
            for path in year_files:
                market = path.parent.name.split("_", 1)[0]
                file_symbols = [symbol for symbol in symbol_list if symbol.partition(".")[0] == market]
                streams.append(self._iter_file(
                    path=path,
                    start_date=max(start_dt.date().isoformat(), f"{year}-01-01"),
                    end_date=min(end_dt.date().isoformat(), f"{year}-12-31"),
                    symbols=file_symbols,
                    frequency=frequency,
                    skip_zero_volume=skip_zero_volume,
                    skip_invalid_ohlc=skip_invalid_ohlc,
                ))
            try:
                for bar in heapq.merge(*streams, key=lambda bar: (bar.bob, bar.symbol)):
                    count += 1
                    # 每次读取只打印前 5 条，查看实际行情内容，避免全量数据刷屏。
                    if count <= 5:
                        print(
                            f"[行情 {count}] 日期={bar.bob.date()} 代码={bar.symbol} "
                            f"开={bar.open} 高={bar.high} 低={bar.low} 收={bar.close} "
                            f"成交量={bar.volume} 成交额={bar.amount}\n"
                            f"  来源文件={bar.extra['source_file']}",
                            flush=True,
                        )
                    yield bar
            finally:
                for stream in streams:
                    stream.close()
        if not count:
            self.logger.warning("No matching daily bars in %s", self.day_bar_dir)

    def iter_batches(self, *, batch_size: int = 10_000, **history_kwargs: Any) -> Iterator[list[BarData]]:
        """Yield bounded lists for vectorized or persistence-oriented consumers."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        batch: list[BarData] = []
        with closing(self.iter_history(**history_kwargs)) as bars:
            for bar in bars:
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
                f"Parquet day-bar years are missing under {self.day_bar_dir}: {display}"
            )

        if missing:
            self.logger.warning("[日线/缺文件] 已允许跳过=%s", missing)
        # Keep years adjacent for iter_history's year-by-year merge.
        return [available[key] for key in sorted(requested, key=lambda key: (key[1], key[0])) if key in available]

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
    ) -> Generator[BarData, None, None]:
        query = pl.scan_parquet(path).filter(
            pl.col("trade_date").is_between(pl.lit(start_date), pl.lit(end_date)),
            pl.col("close").is_not_null(),
        )
        if symbols:
            query = query.filter(pl.col("symbol").is_in(symbols))
        if skip_zero_volume:
            query = query.filter(pl.col("volume") > 0)
        if skip_invalid_ohlc:
            query = query.filter(
                pl.col("high") >= pl.col("low"),
                pl.col("open").is_between(pl.col("low"), pl.col("high")),
                pl.col("close").is_between(pl.col("low"), pl.col("high")),
            )
        # Converted tables are not guaranteed to be in trade-date/symbol order.
        # Filter before collecting; only this market/year is materialized.
        frame = query.sort(["trade_date", "symbol"]).collect(engine="streaming")
        for row in frame.iter_rows(named=True):
            yield MarketBar(
                symbol=row["symbol"],
                bob=datetime.fromisoformat(row["trade_date"]).replace(tzinfo=CHINA_TZ),
                open=row["open"], high=row["high"], low=row["low"], close=row["close"],
                volume=row["volume"], amount=row["amount"],
                frequency=frequency, source=BarSource.PARQUET.value,
                extra={
                    "sec_id": row.get("sec_id"),
                    "pre_close": row.get("pre_close"),
                    "position": row.get("Position"),
                    "updated_at": row.get("updated_at"),
                    "ext_data": row.get("ext_data"),
                    "source_file": str(path),
                    **{key: row[key] for key in ("vwap", "market_cap", "industry", "sector", "subindustry") if key in row},
                },
            )

    def _available_files(self) -> dict[tuple[str, int], Path]:
        files: dict[tuple[str, int], Path] = {}
        for path in self.day_bar_dir.glob("*_????/dists_day_bar.parquet"):
            match = DAY_BAR_DIRECTORY_PATTERN.fullmatch(path.parent.name)
            if match:
                files[(match.group("market"), int(match.group("year")))] = path
        return files

    @staticmethod
    def _resolve_day_bar_dir(root: Path) -> Path:
        candidates = (
            root,
            root / "day_bar",
            root / "basic_data" / "day_bar",
            root / "parquet" / "basic_data" / "day_bar",
        )
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*_????/dists_day_bar.parquet")):
                return candidate
        raise FileNotFoundError(f"Parquet day_bar directory was not found under: {root}")

    @staticmethod
    def _normalize_values(values: str | Iterable[str] | None) -> list[str]:
        if values is None:
            return []
        candidates = values.split(",") if isinstance(values, str) else values
        return [str(value).strip().upper() for value in candidates if str(value).strip()]

    @classmethod
    def _resolve_markets(cls, symbols: list[str], markets: str | Iterable[str] | None) -> list[str]:
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

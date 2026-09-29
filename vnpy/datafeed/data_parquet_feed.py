"""从本地年度 Parquet 文件读取 GM 日线数据。"""

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
from vnpy.datafeed.data_model import BarData, BarSource, MarketBar


# 年度目录格式，例如 SHSE_2025、SZSE_2025。
DAY_BAR_DIRECTORY_PATTERN = re.compile(
    r"^(?P<market>[A-Z]+)_(?P<year>\d{4})$"
)
CHINA_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_STOCK_MARKETS = ("SHSE", "SZSE")


class ParquetDataFeed:
    """从本地年度 Parquet 文件读取日线数据"""

    def __init__(self, root: str | Path) -> None:
        """初始化数据根目录，并定位实际存放日线文件的目录"""
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
        """按时间顺序逐条返回日线；每次只加载一个年份的数据。"""
        if frequency != "1d":
            raise ValueError("Parquet day_bar files only support frequency='1d'")

        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        if start_dt > end_dt:
            raise ValueError("start must not be later than end")

        symbol_list = self._normalize_values(symbols)
        market_list = self._resolve_markets(symbol_list, markets)

        # 根据时间范围和市场，找到需要读取的年度文件。
        files = self.resolve_files(
            start=start_dt,
            end=end_dt,
            markets=market_list,
            allow_missing_years=allow_missing_years,
        )

        count = 0

        # 文件已按年份排列。同一年不同市场的数据分别读取，再合并排序
        for year, year_files in groupby(files, key=lambda path: path.parent.name[-4:],):
            streams: list[Generator[BarData, None, None]] = []

            for path in year_files:
                market = path.parent.name.split("_", 1)[0]

                # 当前文件只接收属于该市场的股票代码。
                file_symbols = [
                    symbol
                    for symbol in symbol_list
                    if symbol.partition(".")[0] == market
                ]

                streams.append(
                    self._iter_file(
                        path=path,
                        start_date=max(
                            start_dt.date().isoformat(),
                            f"{year}-01-01",
                        ),
                        end_date=min(
                            end_dt.date().isoformat(),
                            f"{year}-12-31",
                        ),
                        symbols=file_symbols,
                        frequency=frequency,
                        skip_zero_volume=skip_zero_volume,
                        skip_invalid_ohlc=skip_invalid_ohlc,
                    )
                )

            try:
                # 各市场文件内部已排序；堆归并后按时间、代码依次返回。
                for bar in heapq.merge(*streams, key=lambda bar: (bar.bob, bar.symbol),):
                    count += 1
                    yield bar
            finally:
                # 调用方提前停止迭代时，也关闭尚未读完的生成器。
                for stream in streams:
                    stream.close()

        if not count:
            self.logger.warning("No matching daily bars in %s", self.day_bar_dir,)

    def iter_batches(self, *, batch_size: int = 10_000, **history_kwargs: Any,) -> Iterator[list[BarData]]:
        """将逐条读取的日线按指定数量分批返回"""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        batch: list[BarData] = []

        with closing(self.iter_history(**history_kwargs)) as bars:
            for bar in bars:
                batch.append(bar)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            # 返回不足一个完整批次的剩余数据。
            if batch:
                yield batch

    def load_history(self, **history_kwargs: Any,) -> list[BarData]:
        """将查询结果全部转为列表，适合数据量较小的调用方。"""
        return list(self.iter_history(**history_kwargs))

    def load_frame(
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
    ) -> pl.DataFrame:
        """直接读取为供因子引擎使用的 Polars DataFrame。

        在 Parquet 扫描阶段过滤日期、股票和无效数据，并选择需要的列。
        最终结果及后续因子计算的中间结果需要能够放入内存。
        """
        if frequency != "1d":
            raise ValueError("Parquet day_bar files only support frequency='1d'")

        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        if start_dt > end_dt:
            raise ValueError("start must not be later than end")

        symbol_list = self._normalize_values(symbols)
        files = self.resolve_files(
            start=start_dt,
            end=end_dt,
            markets=self._resolve_markets(symbol_list, markets),
            allow_missing_years=allow_missing_years,
        )

        # 因子研究可能使用的行情字段和扩展字段。
        fields = (
            "open", "high", "low", "close", "volume", "amount",
            "turn", "vwap", "market_cap", "industry",
            "sector", "subindustry",
        )

        scans = []

        for path in files:
            query = self._scan_file(
                path=path,
                start_date=start_dt.date().isoformat(),
                end_date=end_dt.date().isoformat(),
                symbols=symbol_list,
                skip_zero_volume=skip_zero_volume,
                skip_invalid_ohlc=skip_invalid_ohlc,
            )

            # 不同年度文件可能缺少部分扩展字段，只选择实际存在的列
            schema = query.collect_schema()
            scans.append(
                query.select(
                    # 将交易日期转为上海时区的 datetime 主键
                    pl.col("trade_date")
                    .str.to_date()
                    .cast(pl.Datetime("us"))
                    .dt.replace_time_zone("Asia/Shanghai")
                    .alias("datetime"),
                    pl.col("symbol").alias("vt_symbol"),
                    *[
                        pl.col(field)
                        for field in fields
                        if field in schema
                    ],
                )
            )

        if not scans:
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime("us", "Asia/Shanghai"),
                    "vt_symbol": pl.String,
                    **{
                        field: pl.Float64
                        for field in fields[:6]
                    },
                }
            )

        # 合并各年度文件；缺失的扩展列补 null，最后按股票和时间排序。
        return (
            pl.concat(scans, how="diagonal_relaxed", parallel=True)
            .sort(["vt_symbol", "datetime"])
            .collect(engine="streaming")
        )

    def resolve_files(
        self,
        *,
        start: str | datetime,
        end: str | datetime,
        markets: str | Iterable[str] | None = None,
        allow_missing_years: bool = False,
    ) -> list[Path]:
        """查找时间范围内各市场的年度文件，并检查缺失年份。"""
        start_dt = self._parse_datetime(start)
        end_dt = self._parse_datetime(end)
        market_list = (
            self._normalize_values(markets)
            or list(DEFAULT_STOCK_MARKETS)
        )

        requested = {
            (market.upper(), year)
            for market in market_list
            for year in range(start_dt.year, end_dt.year + 1)
        }
        available = self._available_files()
        missing = sorted(requested - available.keys())

        if missing and not allow_missing_years:
            display = ", ".join(
                f"{market}_{year}"
                for market, year in missing
            )
            raise FileNotFoundError(
                "Parquet day-bar years are missing under "
                f"{self.day_bar_dir}: {display}"
            )

        # 先按年份、再按市场排序，供 iter_history 逐年归并。
        return [
            available[key]
            for key in sorted(
                requested,
                key=lambda key: (key[1], key[0]),
            )
            if key in available
        ]

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
        """读取单个市场的年度文件，并逐条转换为 MarketBar。"""
        query = self._scan_file(
            path=path,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            skip_zero_volume=skip_zero_volume,
            skip_invalid_ohlc=skip_invalid_ohlc,
        )

        # 先过滤、排序，再将这个市场和年份的数据载入内存。
        frame = (
            query.sort(["trade_date", "symbol"])
            .collect(engine="streaming")
        )

        for row in frame.iter_rows(named=True):
            yield MarketBar(
                symbol=row["symbol"],
                bob=datetime.fromisoformat(
                    row["trade_date"]
                ).replace(tzinfo=CHINA_TZ),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                amount=row["amount"],
                frequency=frequency,
                source=BarSource.PARQUET.value,
                extra={
                    "sec_id": row.get("sec_id"),
                    "pre_close": row.get("pre_close"),
                    "position": row.get("Position"),
                    "updated_at": row.get("updated_at"),
                    "ext_data": row.get("ext_data"),
                    "source_file": str(path),
                    **{
                        key: row[key]
                        for key in (
                            "vwap",
                            "market_cap",
                            "industry",
                            "sector",
                            "subindustry",
                        )
                        if key in row
                    },
                },
            )

    @staticmethod
    def _scan_file(
        *,
        path: Path,
        start_date: str,
        end_date: str,
        symbols: list[str],
        skip_zero_volume: bool,
        skip_invalid_ohlc: bool,
    ) -> pl.LazyFrame:
        """建立单文件的惰性扫描，并应用统一的数据过滤条件。"""
        query = pl.scan_parquet(path).filter(
            pl.col("trade_date").is_between(
                pl.lit(start_date),
                pl.lit(end_date),
            ),
            pl.col("close").is_not_null(),
        )

        if symbols:
            query = query.filter(
                pl.col("symbol").is_in(symbols)
            )

        if skip_zero_volume:
            query = query.filter(
                pl.col("volume") > 0
            )

        if skip_invalid_ohlc:
            query = query.filter(
                pl.col("high") >= pl.col("low"),
                pl.col("open").is_between(
                    pl.col("low"),
                    pl.col("high"),
                ),
                pl.col("close").is_between(
                    pl.col("low"),
                    pl.col("high"),
                ),
            )

        return query

    def _available_files(self) -> dict[tuple[str, int], Path]:
        """扫描日线目录，建立（市场、年份）到文件路径的映射。"""
        files: dict[tuple[str, int], Path] = {}

        for path in self.day_bar_dir.glob(
            "*_????/dists_day_bar.parquet"
        ):
            match = DAY_BAR_DIRECTORY_PATTERN.fullmatch(
                path.parent.name
            )
            if match:
                files[
                    (
                        match.group("market"),
                        int(match.group("year")),
                    )
                ] = path

        return files

    @staticmethod
    def _resolve_day_bar_dir(root: Path) -> Path:
        """在常见目录结构中定位包含年度文件的 day_bar 目录。"""
        candidates = (
            root,
            root / "day_bar",
            root / "basic_data" / "day_bar",
            root / "parquet" / "basic_data" / "day_bar",
        )

        for candidate in candidates:
            if candidate.is_dir() and any(
                candidate.glob("*_????/dists_day_bar.parquet")
            ):
                return candidate

        raise FileNotFoundError(
            f"Parquet day_bar directory was not found under: {root}"
        )

    @staticmethod
    def _normalize_values(
        values: str | Iterable[str] | None,
    ) -> list[str]:
        """将逗号分隔字符串或可迭代对象规范为大写字符串列表。"""
        if values is None:
            return []

        candidates = (
            values.split(",")
            if isinstance(values, str)
            else values
        )
        return [
            str(value).strip().upper()
            for value in candidates
            if str(value).strip()
        ]

    @classmethod
    def _resolve_markets(
        cls,
        symbols: list[str],
        markets: str | Iterable[str] | None,
    ) -> list[str]:
        """根据股票代码和显式配置确定需要读取的市场。"""
        configured = cls._normalize_values(markets)
        symbol_markets = sorted({
            symbol.partition(".")[0]
            for symbol in symbols
        })

        if configured and symbol_markets:
            invalid = sorted(
                set(symbol_markets) - set(configured)
            )
            if invalid:
                raise ValueError(
                    "symbol markets are not enabled by markets: "
                    f"{', '.join(invalid)}"
                )

        return (
            symbol_markets
            or configured
            or list(DEFAULT_STOCK_MARKETS)
        )

    @staticmethod
    def _parse_datetime(value: str | datetime) -> datetime:
        """将 ISO 格式日期字符串转换为 datetime。"""
        if isinstance(value, datetime):
            return value

        return datetime.fromisoformat(value)
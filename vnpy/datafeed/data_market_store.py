"""模块持有的历史行情快照；计算轮次只借用数据，不决定其生命周期。"""

from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

import polars as pl

from vnpy.config.runtime_config import ParquetConfig
from .data_parquet_feed import ParquetDataFeed


@dataclass(frozen=True)
class MarketDataRequest:
    root: str
    start: str
    end: str
    symbols: tuple[str, ...]
    markets: tuple[str, ...]
    frequency: str
    skip_zero_volume: bool
    skip_invalid_ohlc: bool
    allow_missing_years: bool

    @classmethod
    def from_config(cls, config: ParquetConfig) -> "MarketDataRequest":
        # Only data selection belongs in the cache key, not formulas or batch sizes.
        return cls(
            root=str(Path(config.root).expanduser().resolve()),
            start=config.start, end=config.end,
            symbols=tuple(sorted(set(ParquetDataFeed._normalize_values(config.symbols)))),
            markets=tuple(sorted(set(ParquetDataFeed._normalize_values(config.markets)))),
            frequency=config.frequency, skip_zero_volume=config.skip_zero_volume,
            skip_invalid_ohlc=config.skip_invalid_ohlc,
            allow_missing_years=config.allow_missing_years,
        )


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    request: MarketDataRequest
    frame: pl.DataFrame
    load_seconds: float


@dataclass(frozen=True)
class MarketLoadResult:
    snapshot: MarketSnapshot
    reused: bool


class MarketDataStore:
    """持有一份当前行情快照，刷新成功后整体替换，失败时保留旧快照。

    快照只包含行情。对外返回廉价的 DataFrame clone, 共享底层列缓冲区
    同时避免调用方的列替换等原地操作修改模块持有的 DataFrame 对象。
    """

    def __init__(self) -> None:
        self._snapshot: MarketSnapshot | None = None
        # 写入者串行化；读取快照不必等待磁盘 I/O。
        self._load_lock = RLock()
        self._snapshot_lock = RLock()

    def snapshot(self) -> MarketSnapshot:
        with self._snapshot_lock:
            if self._snapshot is None:
                raise RuntimeError("market data has not been loaded")
            return replace(self._snapshot, frame=self._snapshot.frame.clone())

    def load(self, request: MarketDataRequest, *, reload: bool = False) -> MarketLoadResult:
        # 固定锁顺序：load_lock -> snapshot_lock；读者只获取后者。
        with self._load_lock:
            with self._snapshot_lock:
                if not reload and self._snapshot is not None and self._snapshot.request == request:
                    return MarketLoadResult(self.snapshot(), reused=True)
            # 慢操作不持有快照锁，刷新期间读者仍可取得旧版本。
            started = perf_counter()
            frame = ParquetDataFeed(request.root).load_frame(
                start=request.start, end=request.end, symbols=request.symbols,
                markets=request.markets, frequency=request.frequency,
                skip_zero_volume=request.skip_zero_volume,
                skip_invalid_ohlc=request.skip_invalid_ohlc,
                allow_missing_years=request.allow_missing_years,
            )
            if frame.is_empty():
                raise ValueError("no market data available for history batch")
            if frame["datetime"].null_count() or frame["vt_symbol"].null_count():
                raise ValueError("market input keys must not be null")
            if frame.select(pl.struct("datetime", "vt_symbol").is_duplicated().any()).item():
                raise ValueError("market input contains duplicate datetime/symbol rows")
            with self._snapshot_lock:
                self._snapshot = MarketSnapshot(uuid4().hex, request, frame, perf_counter() - started)
                return MarketLoadResult(self.snapshot(), reused=False)

    def clear(self) -> None:
        """释放模块持有的引用；正在计算的调用方仍可使用其已取得的快照。"""
        # 等待已开始的加载完成，再清空，避免清空后被旧加载重新填入。
        with self._load_lock:
            with self._snapshot_lock:
                self._snapshot = None

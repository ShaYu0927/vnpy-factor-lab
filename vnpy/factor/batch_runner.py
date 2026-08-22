from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

import polars as pl

from vnpy.config.runtime_config import GmSqliteBatchConfig
from vnpy.common.logger import get_logger
from vnpy.datafeed.gm_sqlite_datafeed import GmSqliteDataFeed
from vnpy.datafeed.model import BarData
from vnpy.factor.core.factor_engine import (
    ExecutionMode,
    Factor,
    FactorBatchResult,
    FactorContext,
    FactorEngine,
    FactorStatus,
    FactorValue,
)
from vnpy.factor.realtime_service import BasicFactorSet
from vnpy.factor.kline_report import write_kline_data, write_kline_report
from vnpy.factor.model_pipeline import (
    FactorObservation,
    collect_observations,
)
from vnpy.alpha.modeling.module import model_module_entry
from vnpy.alpha.dataset.datasets import Basic3DL
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType


logger = get_logger("factor.batch_runner")

@dataclass(slots=True)
class BatchFactorSummary:
    """Summary statistics produced by one historical factor batch run."""

    # 从历史数据源累计读取的 K 线总数。
    bars_read: int = 0

    # 已遍历的交易日数量，包含仅用于滚动窗口预热的交易日。
    dates_processed: int = 0

    # 截至当前批次在历史数据中见过的不同证券数量。
    symbols_seen: int = 0

    # 已写入输出端的 FactorValue 记录数，即日期×证券×因子的有效结果数。
    values_written: int = 0

    # FactorEngine 在计算过程中捕获的因子异常数量。
    factor_errors: int = 0

    # 模型训练集与验证集合并后的有效标签样本数量。
    model_training_samples: int = 0

    # 按时间留出的模型测试集有效标签样本数量。
    model_test_samples: int = 0


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

    def __init__(self, path: str | Path, *, overwrite: bool = False, append: bool = False,) -> None:
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
            raise ValueError(f"batch factor output has an incompatible header: {self.path}")

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
        self._latest_values.extend(value for value in rows if (value.trade_date or "") == latest_date)
        return len(rows)

    def close(self) -> None:
        pass

    def __enter__(self) -> ConsoleFactorResultWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def iter_bars_by_date(batches: Iterable[Sequence[BarData]],) -> Iterator[tuple[date, list[BarData]]]:
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
        raise ValueError(f"unknown batch factors: {', '.join(unknown)}; supported: {supported}")
    return tuple(available[name] for name in names)


def calculate_basic_qlib_factors(
    symbol_data_map: dict[str, list[BarData]],
    feature_names: Sequence[str],
    trade_date: date,
) -> FactorBatchResult:
    """Calculate the configured basic factors as one Qlib-style matrix."""
    rows = [
        {
            "datetime": bar.datetime,
            "vt_symbol": symbol,
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        for symbol, bars in symbol_data_map.items()
        for bar in bars
    ]
    matrix = Basic3DL(feature_names=feature_names).load(pl.DataFrame(rows))
    latest = (
        matrix
        .sort(["vt_symbol", "datetime"])
        .group_by("vt_symbol", maintain_order=True)
        .tail(1)
    )

    values: list[FactorValue] = []
    for row in latest.iter_rows(named=True):
        symbol = str(row["vt_symbol"])
        for name in feature_names:
            raw_value = row[name]
            value = float(raw_value) if raw_value is not None else None
            if value is not None and not math.isfinite(value):
                value = None
            values.append(FactorValue(
                symbol=symbol,
                factor_name=name,
                value=value,
                trade_date=trade_date.isoformat(),
                status=(
                    FactorStatus.READY
                    if value is not None
                    else FactorStatus.INVALID
                ),
                reason="" if value is not None else "Qlib expression returned no finite value",
                version="qlib-v1",
            ))
    return FactorBatchResult(values=values)


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
    symbol_text = config.symbols or "<all symbols>"
    symbol_count = len([item for item in symbol_text.split(",") if item])
    logger.info(
        "批处理启动 | 股票 %s 只 | 周期 %s | %s 至 %s",
        symbol_count,
        config.frequency,
        config.start,
        config.end,
    )
    logger.info("股票池 | %s", symbol_text)
    logger.info("数据源 | %s", config.root)
    logger.info(
        "因子配置 | %s | 窗口 %s | %s/%s线程",
        ",".join(config.factors),
        config.window_size,
        config.factor_mode,
        config.max_workers,
    )
    logger.info("结果输出 | %s", config.output)
    factors = select_factors(config.factors)
    use_qlib_loader = set(config.factors).issubset(
        Basic3DL.supported_features()
    )
    if use_qlib_loader:
        logger.info("因子计算 | Qlib fields/names 批量特征矩阵")
    required_bars = max(factor.min_bars for factor in factors)
    if config.window_size < required_bars:
        raise ValueError(
            f"window_size={config.window_size} is too small for selected factors; "
            f"at least {required_bars} bars are required"
        )

    feed = GmSqliteDataFeed(config.root)
    logger.info("行情目录 | %s", feed.day_bar_dir)
    engine = FactorEngine(
        factors=factors,
        mode=ExecutionMode(config.factor_mode),
        max_workers=config.max_workers,
    )
    windows: dict[str, deque[BarData]] = defaultdict(
        lambda: deque(maxlen=config.window_size)
    )
    chart_windows: dict[str, deque[BarData]] = defaultdict(
        lambda: deque(maxlen=config.chart_bars)
    )
    summary = BatchFactorSummary()
    observations: list[FactorObservation] = []
    resume_date = read_last_trade_date(config.output) if config.resume else None
    if resume_date is not None:
        logger.info("断点续跑 | 已保存至 %s", resume_date)
        configured_end = GmSqliteDataFeed._parse_datetime(config.end).date()
        if resume_date >= configured_end:
            logger.info(
                "无需处理 | 结果已更新至 %s",
                resume_date,
            )
            return summary

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
        with create_result_writer(config.output, overwrite=config.overwrite, resume=config.resume,) as writer:
            for trade_date, day_bars in iter_bars_by_date(batches):
                updated_symbols: set[str] = set()
                for bar in day_bars:
                    windows[bar.symbol].append(bar)
                    if config.chart_output or config.chart_data_output:
                        chart_windows[bar.symbol].append(bar)
                    updated_symbols.add(bar.symbol)

                summary.bars_read += len(day_bars)
                summary.dates_processed += 1
                summary.symbols_seen = len(windows)

                if (
                    summary.dates_processed == 1
                    or summary.dates_processed % config.progress_every_dates == 0
                ):
                    logger.info(
                        "处理进度 | %s | %s日 | 行情%s | 股票%s | 因子%s",
                        trade_date,
                        summary.dates_processed,
                        summary.bars_read,
                        summary.symbols_seen,
                        summary.values_written,
                    )

                if resume_date is not None and trade_date <= resume_date:
                    continue

                symbol_data_map = {
                    symbol: list(windows[symbol])
                    for symbol in updated_symbols
                    if len(windows[symbol]) >= required_bars
                }
                if symbol_data_map:
                    if use_qlib_loader:
                        result = calculate_basic_qlib_factors(
                            symbol_data_map,
                            config.factors,
                            trade_date,
                        )
                    else:
                        result = engine.calculate_many(
                            symbol_data_map,
                            context=FactorContext(trade_date=trade_date.isoformat()),
                        )
                    summary.values_written += writer.write(result.values)
                    summary.factor_errors += len(result.errors)
                    if config.train_model:
                        closes = {
                            bar.symbol: float(bar.close)
                            for bar in day_bars
                            if bar.symbol in symbol_data_map and bar.close is not None
                        }
                        observations.extend(
                            collect_observations(
                                result,
                                closes,
                                config.factors,
                                trade_date,
                            )
                        )
                    if result.errors:
                        first = result.errors[0]
                        raise RuntimeError(
                            f"batch factor calculation failed for {first.symbol}/"
                            f"{first.factor_name}: {first.error}"
                        )

    finally:
        executor = getattr(engine, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

    if config.train_model:
        logger.info("模型训练 | 有效观测 %s", len(observations))
        model_result = train_model_via_event(observations, config)
        summary.model_training_samples = model_result.training_samples
        summary.model_test_samples = model_result.test_samples
    if config.chart_output:
        chart_path = write_kline_report(
            chart_windows,
            config.chart_output,
            auto_open=config.open_chart,
        )
        logger.info("K线图表 | %s", chart_path)
    if config.chart_data_output:
        data_path = write_kline_data(chart_windows, config.chart_data_output)
        logger.info("K线数据 | %s", data_path)
    logger.info(
        "处理完成 | %s日 | 行情%s | 股票%s | 因子%s | 错误%s | 训练%s | 测试%s",
        summary.dates_processed,
        summary.bars_read,
        summary.symbols_seen,
        summary.values_written,
        summary.factor_errors,
        summary.model_training_samples,
        summary.model_test_samples,
    )
    return summary


def train_model_via_event(
    observations: Sequence[FactorObservation],
    config: GmSqliteBatchConfig,
):
    """Run model training through the module queue while keeping batch API sync."""
    responses: list[EngineEvent] = []

    def response_entry(_ctx, event: EngineEvent) -> None:
        if event.event_type in {EventType.MODEL_TRAINED, EventType.MODEL_FAILED}:
            responses.append(event)

    engine = ModuleEngine()
    engine.register_module("research", response_entry)
    engine.register_module("model", model_module_entry)
    engine.start_all()
    try:
        request = EngineEvent(
            event_type=EventType.MODEL_TRAIN_REQUEST,
            source="research",
            data={
                "response_target": "research",
                "observations": observations,
                "feature_names": config.factors,
                "horizon": config.label_horizon,
                "model_output": config.model_output,
                "signal_output": config.signal_output,
                "evaluate_factors": config.evaluate_factors,
                "factor_quantiles": config.factor_quantiles,
                "min_abs_ic": config.min_abs_ic,
                "min_abs_ic_ir": config.min_abs_ic_ir,
            },
        )
        accepted = engine.post_event(
            target="model",
            event=request,
        )
        if not accepted:
            raise RuntimeError("model module rejected training request")

        engine.get_module("model")._queue.join()
        engine.get_module("research")._queue.join()
        if not responses:
            raise RuntimeError("model module completed without a response event")
        response = responses[-1]
        if response.event_type == EventType.MODEL_FAILED:
            raise RuntimeError("model training failed") from response.get("error")
        return response.get("result")
    finally:
        engine.stop_all()

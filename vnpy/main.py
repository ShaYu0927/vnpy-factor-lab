from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gm.api import *

from vnpy.common.logger import init_global_logger
from vnpy.datafeed.bar_cache import convert_gm_bar
from vnpy.datafeed.gm_local_datafeed import GmLocalDataFeed
from vnpy.datafeed.gm_sqlite_datafeed import GmSqliteDataFeed
from vnpy.datafeed.model import BarData, BarSource, normalize_bar
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_module import factor_module_entry
from vnpy.config.runtime_config import DEFAULT_BACKTEST_END_TIME, DEFAULT_BACKTEST_START_TIME, DEFAULT_COMMISSION_RATIO, DEFAULT_FACTOR_MAX_WORKERS, DEFAULT_FACTOR_MODE, DEFAULT_FREQUENCY, DEFAULT_GM_TOKEN, DEFAULT_INITIAL_CASH, DEFAULT_RUNTIME_CONFIG, DEFAULT_SLIPPAGE_RATIO, DEFAULT_STRATEGY_ID, DEFAULT_SUBSCRIPTION_FALLBACK_DAYS, DEFAULT_SUBSCRIPTION_QUERY_DATE, GmSqliteConfig, RunMode, RuntimeConfig, load_runtime_config
from vnpy.strategy.strategy_module import strategy_engine_module_entry
from vnpy.subscription.pool import create_default_pool



module_engine = ModuleEngine()


# =============================================================================
# 模块初始化
# =============================================================================

def setup_modules(frequency: str = DEFAULT_FREQUENCY) -> None:
    register_factor_module(frequency)
    register_strategy_module()

    module_engine.start_all()

def register_factor_module(frequency: str) -> None:
    """
    注册实时因子模块。
    """
    if module_engine.module_exists("factor"):
        return

    module_engine.register_module(
        name="factor",
        entry=factor_module_entry,
        config={
            "frequency": frequency,
            "maxlen": 30000,
            "mode": DEFAULT_FACTOR_MODE,
            "max_workers": DEFAULT_FACTOR_MAX_WORKERS,
            "alphas": [],
            "strategy_module": "strategy",
            "enable_print": False,
            "print_every": 20,
        },
    )

def register_strategy_module() -> None:
    """
    注册策略模块。
    """
    if module_engine.module_exists("strategy"):
        return

    module_engine.register_module(
        name="strategy",
        entry=strategy_engine_module_entry,
        config={
            "strategies": [
                {
                    "name": "factor_signal",
                    "class": "vnpy.strategy.factor_signal_strategy.FactorSignalStrategy",
                    "active": False,
                    "factors": [],
                    "setting": {
                        "enable_log": False,
                        "enable_print": False,
                    },
                },
                {
                    "name": "factor_debug",
                    "class": "vnpy.strategy.strategies.factor_debug.factor_debug_strategy.FactorDebugStrategy",
                    "active": False,
                    "factors": [],
                    "setting": {
                        "print_limit": 20,
                        "print_factor_values": False,
                        "max_factor_values": 10,
                    },
                },
            ],
        },
    )


# =============================================================================
# GM 回调函数
# =============================================================================

def init(context) -> None:
    """
    GM 初始化回调

    这里负责：
    1. 启动模块系统；
    2. 创建股票池；
    3. 订阅K线。
    """
    setup_modules(frequency=DEFAULT_FREQUENCY)

    pool = create_default_pool(
        query_date=DEFAULT_SUBSCRIPTION_QUERY_DATE,
        fallback_days=DEFAULT_SUBSCRIPTION_FALLBACK_DAYS,
    )
    symbol_list = pool.symbols()
    symbol_list = symbol_list[:5]

    if not symbol_list:
        return

    symbols = ",".join(symbol_list)

    subscribe(symbols=symbols, frequency=DEFAULT_FREQUENCY, count=30,)


def on_bar(context, bars) -> None:
    """
    GM K线回调
    """
    converted_bars = [
        convert_gm_bar(raw_bar, frequency=DEFAULT_FREQUENCY)
        for raw_bar in bars
    ]

    for bar in converted_bars:
        post_bar(bar, source=BarSource.GM_LIVE.value)


def algo(context) -> None:
    """
    GM algo 回调。
    """
    if hasattr(context, "strategy"):
        context.strategy.on_bar(context)


def post_bar(bar: BarData, source: str) -> bool:
    bar = normalize_bar(bar, source=source)
    return module_engine.post_event(
        target="factor",
        event=EngineEvent(
            event_type=EventType.BAR,
            source=source,
            symbol=bar.symbol,
            data={"bar": bar},
        ),
    )


def wait_module_idle(name: str) -> None:
    node = module_engine.get_module(name)
    if node is None:
        return
    node._queue.join()

def run_db_replay(db_path: str | Path, frequency: str = DEFAULT_FREQUENCY, symbols: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,) -> None:
    setup_modules(frequency=frequency)
    symbol_list = parse_symbols(symbols)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        sql, params = build_bar_query(frequency=frequency, symbols=symbol_list, start=start, end=end,)
        count = 0

        for row in conn.execute(sql, params):
            bar = row_to_bar(row)
            post_bar(bar, source=BarSource.SQLITE.value)

        wait_module_idle("factor")
        wait_module_idle("strategy")


    finally:
        conn.close()
        module_engine.stop_all()


def run_gm_local_replay(
    symbols: str,
    frequency: str = DEFAULT_FREQUENCY,
    start: Optional[str] = None,
    end: Optional[str] = None,
    count: Optional[int] = None,
) -> None:
    """Replay downloaded GM bars through the normal module event pipeline."""
    feed = GmLocalDataFeed(token=DEFAULT_GM_TOKEN)

    if count is not None:
        bars = feed.load_recent(
            symbols=symbols,
            frequency=frequency,
            count=count,
            end=end,
        )
    else:
        if not start or not end:
            raise ValueError("--gm-local requires --start and --end, or --count")
        bars = feed.load_history(
            symbols=symbols,
            frequency=frequency,
            start=start,
            end=end,
        )

    setup_modules(frequency=frequency)
    print_gm_local_summary(bars, frequency)

    try:
        for index, bar in enumerate(bars, start=1):
            post_bar(bar, source=BarSource.GM_LOCAL.value)

        wait_module_idle("factor")
        wait_module_idle("strategy")
    finally:
        module_engine.stop_all()


def run_gm_sqlite_replay(config: GmSqliteConfig) -> None:
    """Stream GM yearly SQLite bars through the existing factor pipeline."""
    feed = GmSqliteDataFeed(config.root)
    setup_modules(frequency=config.frequency)
    symbols = parse_symbols(config.symbols)
    first_bar: BarData | None = None
    last_bar: BarData | None = None
    symbol_set: set[str] = set()
    count = 0

    try:
        bars = feed.iter_history(
            start=config.start,
            end=config.end,
            symbols=symbols,
            markets=config.markets,
            frequency=config.frequency,
            skip_zero_volume=config.skip_zero_volume,
            skip_invalid_ohlc=config.skip_invalid_ohlc,
            allow_missing_years=config.allow_missing_years,
        )
        for bar in bars:
            while module_engine.queue_size("factor") >= config.max_inflight:
                time.sleep(0.005)

            if not post_bar(bar, source=BarSource.GM_SQLITE.value):
                raise RuntimeError("factor queue rejected a GM SQLite bar")

            count += 1
            first_bar = first_bar or bar
            last_bar = bar
            symbol_set.add(bar.symbol)

        wait_module_idle("factor")
        wait_module_idle("strategy")
    finally:
        module_engine.stop_all()


def print_gm_local_summary(bars: List[BarData], frequency: str) -> None:
    if not bars:
        return

    counts = Counter(bar.symbol for bar in bars)
    symbol_counts = ", ".join(
        f"{symbol}:{count}" for symbol, count in sorted(counts.items())
    )
    first_bob = min(bar.bob for bar in bars)
    last_bob = max(bar.bob for bar in bars)



def parse_symbols(symbols: Optional[str]) -> List[str]:
    if not symbols:
        return []

    return [item.strip() for item in symbols.split(",") if item.strip()]


def build_bar_query(frequency: str,  symbols: List[str], start: Optional[str], end: Optional[str],) -> tuple[str, list]:
    params: list = [frequency]

    if symbols:
        placeholders = ",".join(["?"] * len(symbols))
        sql += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    if start:
        sql += " AND bob >= ?"
        params.append(start)

    if end:
        sql += " AND bob <= ?"
        params.append(end)

    sql += " ORDER BY bob ASC, symbol ASC"

    return sql, params


def row_to_bar(row: sqlite3.Row) -> BarData:
    return normalize_bar(dict(row),source=BarSource.SQLITE,)


def parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value

    return datetime.fromisoformat(str(value))

def run_gm_backtest(config=None) -> None:
    """
    启动 GM 回测。
    """
    strategy_id = config.strategy_id if config else DEFAULT_STRATEGY_ID
    start = config.start if config else DEFAULT_BACKTEST_START_TIME
    end = config.end if config else DEFAULT_BACKTEST_END_TIME
    initial_cash = config.initial_cash if config else DEFAULT_INITIAL_CASH
    commission_ratio = config.commission_ratio if config else DEFAULT_COMMISSION_RATIO
    slippage_ratio = config.slippage_ratio if config else DEFAULT_SLIPPAGE_RATIO

    run(
        strategy_id=strategy_id,
        filename="main.py",
        mode=MODE_BACKTEST,
        token=DEFAULT_GM_TOKEN,
        backtest_start_time=start,
        backtest_end_time=end,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=initial_cash,
        backtest_commission_ratio=commission_ratio,
        backtest_slippage_ratio=slippage_ratio,
    )


def init_logger() -> None:
    init_global_logger(
        app_name="quant",
        log_dir="logs",
        level=20,
        max_bytes=20 * 1024 * 1024,
        backup_count=20,
        enable_console=False,
        enable_file=True,
    )


def run_from_config(config: RuntimeConfig) -> None:
    if config.mode == RunMode.GM_LOCAL:
        setting = config.gm_local
        assert setting is not None
        run_gm_local_replay(symbols=setting.symbols, frequency=setting.frequency, start=setting.start, end=setting.end, count=setting.count,)
        return

    if config.mode == RunMode.GM_SQLITE:
        setting = config.gm_sqlite
        assert setting is not None
        run_gm_sqlite_replay(setting)
        return

    if config.mode == RunMode.DATABASE:
        setting = config.database
        assert setting is not None
        run_db_replay(db_path=setting.path, frequency=setting.frequency, symbols=setting.symbols, start=setting.start, end=setting.end,)
        return

    run_gm_backtest(config.gm_backtest)


def main() -> None:
    init_logger()
    config = load_runtime_config(DEFAULT_RUNTIME_CONFIG)
    run_from_config(config)


if __name__ == "__main__":
    main()

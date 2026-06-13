from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gm.api import *

from vnpy.common.logger import init_global_logger
from vnpy.datafeed.BarCache import BarData, convert_gm_bar
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_module import factor_module_entry
from vnpy.strategy.StratModule import strategy_engine_module_entry
from vnpy.subscription.pool import create_default_pool


# =============================================================================
# 全局配置
# =============================================================================

DEFAULT_FREQUENCY = "60s"

DEFAULT_STRATEGY_ID = "a2c12b21-3191-11f1-9539-fa89d2391347"

DEFAULT_BACKTEST_START_TIME = "2026-03-01 08:00:00"
DEFAULT_BACKTEST_END_TIME = "2026-04-30 16:00:00"

DEFAULT_INITIAL_CASH = 10_000_000
DEFAULT_COMMISSION_RATIO = 0.0001
DEFAULT_SLIPPAGE_RATIO = 0.0001

EVENT_ML_SIGNAL = "eMlSignal"

module_engine = ModuleEngine()


# =============================================================================
# 模块初始化
# =============================================================================

def setup_modules(frequency: str = DEFAULT_FREQUENCY) -> None:
    """
    初始化因子模块和策略模块。
    """
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
            "strategy_module": "strategy",
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
                    "active": True,
                    "factors": ["momentum_20", "volatility_20", "volume_20"],
                    "setting": {
                        "enable_log": True,
                        "enable_print": True,
                    },
                },
                {
                    "name": "factor_debug",
                    "class": "vnpy.strategy.strategies.FactorDebugStrategy.factor_debug_strategy.FactorDebugStrategy",
                    "active": True,
                    "factors": ["momentum_20", "volatility_20", "volume_20"],
                    "setting": {
                        "print_limit": 20,
                        "print_factor_values": True,
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
    GM 初始化回调。

    这里负责：
    1. 启动模块系统；
    2. 创建股票池；
    3. 订阅K线。
    """
    setup_modules(frequency=DEFAULT_FREQUENCY)

    pool = create_default_pool()
    symbol_list = pool.symbols()

    # 调试阶段建议先限制数量，避免一次订阅 5000 多只。
    symbol_list = symbol_list[:5]

    symbols = ",".join(symbol_list)

    print(
        f"[main.init] subscribe symbols count={len(symbol_list)}, "
        f"symbols={symbols}",
        flush=True,
    )

    subscribe(
        symbols=symbols,
        frequency=DEFAULT_FREQUENCY,
        count=30,
    )

    print("[main.init] subscribe called", flush=True)


def on_bar(context, bars) -> None:
    """
    GM K线回调。
    """
    print(f"[main.on_bar] received bars, count={len(bars)}", flush=True)

    for raw_bar in bars:
        bar = convert_gm_bar(raw_bar, frequency=DEFAULT_FREQUENCY)

        print(
            f"[main.on_bar] symbol={bar.symbol}, "
            f"bob={bar.bob}, "
            f"close={bar.close}, "
            f"volume={bar.volume}",
            flush=True,
        )

        post_bar(bar, source="gm")


def algo(context) -> None:
    """
    GM algo 回调。
    """
    if hasattr(context, "strategy"):
        context.strategy.on_bar(context)


# =============================================================================
# 事件投递
# =============================================================================

def post_bar(bar: BarData, source: str) -> bool:
    """
    将K线投递给因子模块。
    """
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
    """
    等待指定模块队列处理完成。
    """
    node = module_engine.get_module(name)
    if node is None:
        return

    node._queue.join()


# =============================================================================
# 数据库回放
# =============================================================================

def run_db_replay(
    db_path: str | Path,
    frequency: str = DEFAULT_FREQUENCY,
    symbols: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    """
    从 SQLite 数据库读取K线，并回放给模块系统。

    表结构默认使用 bar_data：
        symbol, bob, open, high, low, close, volume, amount, frequency
    """
    setup_modules(frequency=frequency)

    symbol_list = parse_symbols(symbols)

    print(
        f"[DB] replay start, db_path={db_path}, "
        f"frequency={frequency}, symbols={symbol_list}, "
        f"start={start}, end={end}",
        flush=True,
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        sql, params = build_bar_query(
            frequency=frequency,
            symbols=symbol_list,
            start=start,
            end=end,
        )

        print(f"[DB] query sql={sql}", flush=True)
        print(f"[DB] query params={params}", flush=True)

        count = 0

        for row in conn.execute(sql, params):
            bar = row_to_bar(row)
            post_bar(bar, source="db")

            count += 1

            if count % 1000 == 0:
                print(
                    f"[DB] replaying, count={count}, "
                    f"last_symbol={bar.symbol}, last_bob={bar.bob}",
                    flush=True,
                )

        wait_module_idle("factor")
        wait_module_idle("strategy")

        print(f"[DB] replay finished, total_count={count}", flush=True)

    finally:
        conn.close()
        module_engine.stop_all()


def parse_symbols(symbols: Optional[str]) -> List[str]:
    """
    解析命令行传入的 symbols 参数。
    """
    if not symbols:
        return []

    return [item.strip() for item in symbols.split(",") if item.strip()]


def build_bar_query(
    frequency: str,
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
) -> tuple[str, list]:
    """
    构造数据库查询 SQL。
    """
    sql = """
        SELECT symbol, bob, open, high, low, close, volume, amount, frequency
        FROM bar_data
        WHERE frequency = ?
    """

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
    """
    将数据库行转换为 BarData。
    """
    return BarData(
        symbol=str(row["symbol"]),
        bob=parse_datetime(row["bob"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        amount=None if row["amount"] is None else float(row["amount"]),
        frequency=str(row["frequency"]),
    )


def parse_datetime(value) -> datetime:
    """
    兼容数据库里的时间格式。
    """
    if isinstance(value, datetime):
        return value

    return datetime.fromisoformat(str(value))



def run_gm_backtest() -> None:
    """
    启动 GM 回测。
    """
    token = os.getenv("GM_TOKEN", "")

    if not token:
        token = "ad3b5bc0baaf82a4572f36cff8242f448063e439"

    run(
        strategy_id=DEFAULT_STRATEGY_ID,
        filename="main.py",
        mode=MODE_BACKTEST,
        token=token,
        backtest_start_time=DEFAULT_BACKTEST_START_TIME,
        backtest_end_time=DEFAULT_BACKTEST_END_TIME,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=DEFAULT_INITIAL_CASH,
        backtest_commission_ratio=DEFAULT_COMMISSION_RATIO,
        backtest_slippage_ratio=DEFAULT_SLIPPAGE_RATIO,
    )


# =============================================================================
# 日志 / 参数 / 主入口
# =============================================================================

def init_logger() -> None:
    """
    初始化全局日志。
    """
    init_global_logger(
        app_name="quant",
        log_dir="logs",
        level=20,
        max_bytes=20 * 1024 * 1024,
        backup_count=20,
        enable_console=True,
        enable_file=True,
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Run quant modules with GM or database data."
    )

    parser.add_argument("--db", help="sqlite database file path for bar replay")
    parser.add_argument("--frequency", default=DEFAULT_FREQUENCY, help="bar frequency")
    parser.add_argument("--symbols", help="symbols, comma separated")
    parser.add_argument("--start", help="start datetime, example: 2026-03-01 09:30:00")
    parser.add_argument("--end", help="end datetime, example: 2026-03-31 15:00:00")

    return parser.parse_args()


def main() -> None:

    init_logger()

    args = parse_args()

    if args.db:
        run_db_replay(
            db_path=args.db,
            frequency=args.frequency,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
        )
        return

    run_gm_backtest()


if __name__ == "__main__":
    main()
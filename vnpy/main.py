from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from vnpy.common.logger import init_global_logger, get_logger, shutdown_global_logger
from vnpy.datafeed.parquet_datafeed import ParquetDataFeed
from vnpy.datafeed.model import BarData, BarSource
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_module import factor_module_entry
from vnpy.config.runtime_config import (
    DEFAULT_FACTOR_MAX_WORKERS,
    DEFAULT_FACTOR_MODE,
    DEFAULT_FREQUENCY,
    DEFAULT_RUNTIME_CONFIG,
    ParquetConfig,
    RunMode,
    RuntimeConfig,
    load_runtime_config,
)
from vnpy.strategy.strategy_module import strategy_engine_module_entry


module_engine = ModuleEngine()
logger = get_logger("main")


# =============================================================================
# 模块初始化
# =============================================================================

def setup_modules(frequency: str = DEFAULT_FREQUENCY, alphas=None, universe=None) -> None:
    logger.info("[main/modules] starting factor and strategy modules frequency=%s", frequency)
    register_factor_module(frequency, alphas=alphas, universe=universe)
    register_strategy_module()

    module_engine.start_all()

def register_factor_module(frequency: str, alphas=None, universe=None) -> None:
    """
    注册实时因子模块。
    """
    if module_engine.module_exists("factor"):
        return
    if not alphas:
        logger.warning("[main/factor] alphas=[]: replay will not calculate factors")

    module_engine.register_module(
        name="factor",
        entry=factor_module_entry,
        config={
            "frequency": frequency,
            "maxlen": 30000,
            "mode": DEFAULT_FACTOR_MODE,
            "max_workers": DEFAULT_FACTOR_MAX_WORKERS,
            "alphas": alphas or [],
            "universe": universe,
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


def post_bar(bar: BarData, source: str) -> bool:
    bar.source = source
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

def run_parquet_replay(config: ParquetConfig, alphas=None) -> None:
    """
    将本地 Parquet 历史行情按时间顺序送入因子、策略事件处理流程。
    每根 K 线作为 BAR 事件发送给因子模块，具体计算由模块按配置执行。
    """
    started = time.perf_counter()
    # 创建本地读取器，启动因子和策略模块，并解析股票代码筛选条件。
    feed = ParquetDataFeed(config.root)
    symbols = parse_symbols(config.symbols)
    setup_modules(frequency=config.frequency, alphas=alphas, universe=symbols or None)
    # 记录回放的首尾行情、股票数量和累计条数，用于进度与完成日志。
    first_bar: BarData | None = None
    last_bar: BarData | None = None
    symbol_set: set[str] = set()
    count = 0

    try:
        # 按日期、市场和代码筛选，迭代时按年份读取并按时间、代码顺序输出。
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
            # 队列积压达到上限时暂停发送，每 5 毫秒检查一次，等待下游消化。
            while module_engine.queue_size("factor") >= config.max_inflight:
                time.sleep(0.005)

            # 将当前 K 线包装成 BAR 事件入队；因子计算在模块中异步执行。
            if not post_bar(bar, source=BarSource.PARQUET.value):
                raise RuntimeError("factor queue rejected a Parquet bar")

            count += 1
            first_bar = first_bar or bar
            last_bar = bar
            symbol_set.add(bar.symbol)
            if count == 1 or count % max(1, config.progress_every) == 0:
                logger.info("[replay/progress] bars=%d symbols=%d latest=%s %s queue=%d elapsed=%.2fs",
                            count, len(symbol_set), bar.symbol, bar.bob,
                            module_engine.queue_size("factor"), time.perf_counter() - started)

        # 文件读完不代表计算完成：先等因子处理完，再等其下游策略处理完。
        logger.info("[replay/drain] read complete bars=%d; waiting for factor/strategy queues", count)
        wait_module_idle("factor")
        wait_module_idle("strategy")
        logger.info("[replay/complete] bars=%d symbols=%d first=%s last=%s elapsed=%.2fs",
                    count, len(symbol_set), first_bar.bob if first_bar else None,
                    last_bar.bob if last_bar else None, time.perf_counter() - started)
        if not count:
            logger.warning("[replay/empty] no matching bars; check source directory, date range, symbols and filters")
    finally:
        # 回放正常结束或处理中发生异常时，都关闭已启动的模块。
        module_engine.stop_all()


def parse_symbols(symbols: str | None) -> list[str]:
    if not symbols:
        return []
    return [item.strip() for item in symbols.split(",") if item.strip()]


def init_logger() -> None:
    init_global_logger(
        app_name="quant",
        log_dir="logs",
        level=20,
        max_bytes=20 * 1024 * 1024,
        backup_count=20,
        enable_console=True,
        enable_file=True,
    )


def run_from_config(config: RuntimeConfig) -> None:
    """Run import, factor calculation, or replay from local Parquet files."""
    if config.mode != RunMode.PARQUET:
        raise ValueError("only local parquet mode is supported")
    setting = config.parquet
    raw_alphas = config.raw.get("alphas", [])
    if not isinstance(raw_alphas, list):
        raise ValueError("alphas must be a list of name/formula objects")
    alpha_engine = None
    if raw_alphas:
        from vnpy.alpha import Alpha, AlphaEngine

        # Parse and validate before starting a potentially expensive import.
        alpha_engine = AlphaEngine([Alpha(**item) for item in raw_alphas])
    import_options = config.raw.get("parquet_import", {})
    if import_options.get("enabled", False):
        from vnpy.factor.parquet_batch_runner import calculate_imported_alphas, import_parquet_history
        snapshot = import_parquet_history(setting, import_options)
        if alpha_engine is not None:
            calculate_imported_alphas(snapshot, alpha_engine)
        return

    if alpha_engine is not None and alpha_engine.requires_cross_section and not parse_symbols(setting.symbols):
        raise ValueError("cross-sectional replay requires an explicit parquet.symbols universe")
    run_parquet_replay(setting, alphas=raw_alphas)


def main() -> None:
    init_logger()
    try:
        config = load_runtime_config(DEFAULT_RUNTIME_CONFIG)
        run_from_config(config)
    finally:
        shutdown_global_logger()


if __name__ == "__main__":
    main()

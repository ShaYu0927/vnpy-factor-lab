from __future__ import annotations

import sys
import time
from pathlib import Path
from itertools import groupby, islice

import polars as pl

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from vnpy.common.logger import init_global_logger, get_logger, shutdown_global_logger
from vnpy.datafeed.data_parquet_feed import ParquetDataFeed
from vnpy.datafeed.data_model import BarData, BarSource
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.factor_realtime_module import factor_module_entry
from vnpy.config.runtime_config import (
    DEFAULT_FREQUENCY,
    DEFAULT_RUNTIME_CONFIG,
    ParquetConfig,
    RunMode,
    RuntimeConfig,
    load_runtime_config,
)
from vnpy.strategy.strategy_module import strategy_engine_module_entry
from vnpy.alpha.engine import AlphaEngine


module_engine = ModuleEngine()
logger = get_logger("main")


# =============================================================================
# 模块初始化
# =============================================================================

def setup_modules(frequency: str = DEFAULT_FREQUENCY, alphas=None, universe=None, optional_alphas=()) -> None:
    logger.debug("[main/modules] starting factor and strategy modules frequency=%s", frequency)
    # A second run in the same interpreter must not retain the previous formulas/cache.
    for name in ("factor", "strategy"):
        if module_engine.is_module_started(name):
            raise RuntimeError(f"cannot reconfigure running module: {name}")
    for name in ("factor", "strategy"):
        if module_engine.module_exists(name):
            module_engine.unregister_module(name)
    register_factor_module(frequency, alphas=alphas, universe=universe, optional_alphas=optional_alphas)
    register_strategy_module()

    module_engine.start_all()

def register_factor_module(frequency: str, alphas=None, universe=None, optional_alphas=()) -> None:
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
            "sample_maxlen": 1000,
            "alphas": alphas or [],
            "optional_alphas": list(optional_alphas),
            "universe": universe,
            "strategy_module": "strategy",
            "enable_print": False,
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
    if node.last_error is not None:
        raise RuntimeError(f"module {name} failed: {node.last_error}") from node.last_error

def run_parquet_replay(config: ParquetConfig, alphas=None, optional_alphas=()) -> dict:
    """
    将本地 Parquet 历史行情按时间顺序送入因子、策略事件处理流程。
    同一时间点的 K 线分批进入因子模块，因子与策略处理完成后推进时间。
    """
    started = time.perf_counter()
    # 创建本地读取器，启动因子和策略模块，并解析股票代码筛选条件。
    feed = ParquetDataFeed(config.root)
    symbols = parse_symbols(config.symbols)
    setup_modules(frequency=config.frequency, alphas=alphas, universe=symbols or None,
                  optional_alphas=optional_alphas)
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
        next_progress = config.progress_every
        for at, same_time in groupby(bars, key=lambda bar: bar.bob):
            while batch := list(islice(same_time, min(config.batch_size, config.max_inflight))):
                # Drain each bounded batch through both modules. This also keeps
                # the strategy queue bounded when factor calculation becomes faster.
                for bar in batch:
                    bar.source = BarSource.PARQUET.value
                    symbol_set.add(bar.symbol)
                if first_bar is None:
                    # 回放不经过全局行情表；把首批 BAR 排成同样的列供检查。
                    preview = pl.DataFrame([{
                        "datetime": bar.bob,
                        "vt_symbol": bar.symbol,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "amount": bar.amount,
                    } for bar in batch[:5]])
                    with pl.Config(tbl_cols=-1, tbl_width_chars=180, fmt_str_lengths=28):
                        display = str(preview)
                    logger.info("[replay/market-preview] first %d BAR rows:\n%s", preview.height, display)
                if not module_engine.post_event("factor", EngineEvent(
                    event_type=EventType.BAR, source=BarSource.PARQUET.value, data={"bars": batch},
                )):
                    raise RuntimeError("factor queue rejected a Parquet batch")
                wait_module_idle("factor")
                wait_module_idle("strategy")
                count += len(batch)
                first_bar = first_bar or batch[0]
                last_bar = batch[-1]
            if count >= next_progress:
                logger.info("[replay/progress] bars=%d symbols=%d latest=%s elapsed=%.2fs",
                            count, len(symbol_set), at, time.perf_counter() - started)
                next_progress = count + config.progress_every

        # 文件读完不代表计算完成：先等因子处理完，再等其下游策略处理完。
        logger.debug("[replay/drain] read complete bars=%d; waiting for factor/strategy queues", count)
        wait_module_idle("factor")
        wait_module_idle("strategy")
        logger.debug("[replay/complete] bars=%d symbols=%d first=%s last=%s elapsed=%.2fs",
                    count, len(symbol_set), first_bar.bob if first_bar else None,
                    last_bar.bob if last_bar else None, time.perf_counter() - started)
        if not count:
            logger.warning("[replay/empty] no matching bars; check source directory, date range, symbols and filters")
        factor_ctx = module_engine.get_context("factor")
        strategy_ctx = module_engine.get_context("strategy")
        summary = {
            "bars": count, "symbols": len(symbol_set),
            "first": first_bar.bob.isoformat() if first_bar else None,
            "last": last_bar.bob.isoformat() if last_bar else None,
            "batches": factor_ctx.get_state("calculation_batches", 0),
            "factor_samples": factor_ctx.get_state("sample_count", 0),
            "strategy_samples": strategy_ctx.get_state("received_samples", 0),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_queues": {name: module_engine.get_module(name).peak_queue_size for name in ("factor", "strategy")},
        }
        logger.info("[replay/complete] %s", summary)
        return summary
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


def run_from_config(config: RuntimeConfig) -> dict | None:
    """根据配置执行全市场批量计算、Parquet 导入或历史数据回放。

    先解析固定公式，按 expression_iteration 配置生成并合并候选。
    启用 history_batch 时直接读取列式行情并计算全部因子。
    启用 parquet_import 时导入行情并计算全部公式；否则将同一组
    公式送入现有因子、策略事件链。未启用表达式迭代时沿用固定公式。
    """
    if config.mode != RunMode.PARQUET:
        raise ValueError("only local parquet mode is supported")
    setting = config.parquet
    from vnpy.factor.factor_history_batch import parse_history_batch_config, run_history_batch
    batch_options = parse_history_batch_config(config.raw.get("history_batch", {}), config.config_dir)
    if batch_options is not None and config.raw.get("parquet_import", {}).get("enabled", False):
        raise ValueError("history_batch and parquet_import cannot both be enabled")
    from vnpy.alpha.modeling.evaluation_config import parse_evaluation_config
    evaluation = parse_evaluation_config(config.raw.get("factor_evaluation", {}), config.config_dir)
    from vnpy.alpha.mining.runtime import prepare_runtime_alphas
    resolved = prepare_runtime_alphas(config)
    raw_alphas = resolved.as_config()
    alpha_engine = None
    if raw_alphas:
        alpha_engine = AlphaEngine(resolved.definitions)
    if evaluation is not None:
        if alpha_engine is None:
            raise ValueError("factor evaluation requires at least one alpha")
        from vnpy.alpha.modeling.runtime_evaluation import run_factor_evaluation
    import_options = config.raw.get("parquet_import", {})
    if batch_options is not None:
        if alpha_engine is None:
            raise ValueError("history batch requires at least one alpha")
        return run_history_batch(config, batch_options, alpha_engine, raw_alphas,
                                 module_engine=module_engine, evaluation=evaluation)
    if import_options.get("enabled", False):
        from vnpy.factor.factor_parquet_batch_runner import calculate_imported_alphas, import_parquet_history
        snapshot = import_parquet_history(setting, import_options)
        if alpha_engine is not None:
            calculate_imported_alphas(snapshot, alpha_engine)
        if evaluation is not None:
            return {"import": snapshot, "evaluation": run_factor_evaluation(
                config, evaluation, alpha_engine, raw_alphas, snapshot=snapshot,
            )}
        return

    if alpha_engine is not None and alpha_engine.requires_cross_section and not parse_symbols(setting.symbols):
        raise ValueError("cross-sectional replay requires an explicit parquet.symbols universe")
    if resolved.generated_names:
        result = run_parquet_replay(setting, alphas=raw_alphas, optional_alphas=resolved.generated_names)
    else:
        result = run_parquet_replay(setting, alphas=raw_alphas)
    if evaluation is not None:
        result["evaluation"] = run_factor_evaluation(config, evaluation, alpha_engine, raw_alphas)
    return result


def main(config_path: str | Path = DEFAULT_RUNTIME_CONFIG) -> None:
    init_logger()
    try:
        config = load_runtime_config(config_path)
        run_from_config(config)
    finally:
        from vnpy.datafeed.data_market_module import unregister_market_module
        try:
            unregister_market_module(module_engine)
        finally:
            shutdown_global_logger()


if __name__ == "__main__":
    main()

"""Read a complete market interval and calculate all configured formulas once."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from time import perf_counter

import polars as pl

from vnpy.common.logger import get_logger
from vnpy.datafeed.data_daily_store import atomic_parquet
from vnpy.datafeed.data_market_module import load_market_snapshot


@dataclass(frozen=True)
class HistoryBatchConfig:
    workers: int = 1
    output_root: str = ""


def parse_history_batch_config(raw, config_dir: Path) -> HistoryBatchConfig | None:
    if not isinstance(raw, dict):
        raise ValueError("history_batch must be an object")
    if set(raw) - {"enabled", "workers", "output_root"}:
        raise ValueError("unknown history_batch option")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("history_batch enabled must be a boolean")
    workers = raw.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("history_batch workers must be a positive integer")
    root = raw.get("output_root", "../data/history_batch")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("history_batch output_root must be a nonempty string")
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return HistoryBatchConfig(workers, str(path.resolve())) if enabled else None


def run_history_batch(runtime, options, engine, definitions, *, module_engine, evaluation=None) -> dict:
    """Borrow module-owned market data for one formula calculation round."""
    started = perf_counter()
    logger = get_logger("main.history_batch")
    setting = runtime.parquet
    logger.info("[batch/market] requesting markets=%s start=%s end=%s", setting.markets, setting.start, setting.end)
    loaded = load_market_snapshot(module_engine, setting)
    market = loaded.snapshot.frame
    read_seconds = perf_counter() - started
    logger.info("[batch/market] snapshot=%s reused=%s", loaded.snapshot.snapshot_id, loaded.reused)
    logger.info("[batch/calculate] rows=%d symbols=%d factors=%d workers=%d polars_threads=%d",
                market.height, market["vt_symbol"].n_unique(), len(engine.definitions),
                options.workers, pl.thread_pool_size())
    calculation_started = perf_counter()
    factors = engine.calculate_parallel(market, workers=options.workers)
    calculate_seconds = perf_counter() - calculation_started
    root = Path(options.output_root)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%SZ_"), dir=root))
    manifest = {
        "status": "running", "input": asdict(setting), "alphas": definitions,
        "config": asdict(options), "runtime": runtime.raw,
        "polars_version": pl.__version__, "polars_threads": pl.thread_pool_size(),
        "market_owner": "ModuleEngine.market.market_store",
        "partitioning": "formula groups; every group sees the complete market interval",
        "warmup": "uses only configured interval; unavailable history remains null",
        "cross_section": "available observations at each date; missing bars are not filled",
    }
    summary = {
        "rows": market.height, "symbols": market["vt_symbol"].n_unique(),
        "dates": market["datetime"].n_unique(), "factors": len(engine.definitions),
        "first": market["datetime"].min().isoformat(), "last": market["datetime"].max().isoformat(),
        "read_seconds": round(read_seconds, 6), "calculate_seconds": round(calculate_seconds, 6),
        "market_snapshot_id": loaded.snapshot.snapshot_id, "market_reused": loaded.reused,
        "market_load_seconds": round(loaded.snapshot.load_seconds, 6),
        "output": str(output), "factors_path": str(output / "factors.parquet"),
    }
    try:
        atomic_parquet(market, output / "market.parquet")
        atomic_parquet(factors, output / "factors.parquet")
        if evaluation is not None:
            from vnpy.alpha.modeling.runtime_evaluation import evaluate_frames
            summary["evaluation"] = evaluate_frames(
                market, factors, definitions, evaluation,
                context={"runtime": runtime.raw, "input": asdict(setting),
                         "factor_source": "history batch result reused without rereading or recalculating"},
            )
        summary["elapsed_seconds"] = round(perf_counter() - started, 6)
        manifest.update(summary, status="complete")
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    logger.info("[batch/complete] %s", summary)
    return summary

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Any

DEFAULT_FREQUENCY = "1d"
DEFAULT_FACTOR_MODE = os.getenv("VNPY_FACTOR_MODE", "thread")
DEFAULT_FACTOR_MAX_WORKERS = int(os.getenv("VNPY_FACTOR_MAX_WORKERS", "4"))
DEFAULT_RUNTIME_CONFIG = Path(__file__).resolve().parents[2] / "config" / "runtime.basic_alphas.json"

EVENT_ML_SIGNAL = "eMlSignal"


class RunMode(str, Enum):
    PARQUET = "parquet"


@dataclass(frozen=True)
class ParquetConfig:
    """Parquet 数据源的读取和处理参数。"""

    root: str                       # Parquet 数据根目录
    start: str                      # 数据起始日期
    end: str                        # 数据结束日期
    symbols: str | None = None      # 股票代码列表；None 表示不按代码筛选
    markets: str = "SHSE,SZSE"       # 要读取的市场
    frequency: str = "1d"           # 数据频率，目前只支持日线
    skip_zero_volume: bool = True   # 跳过成交量为零的记录
    skip_invalid_ohlc: bool = True  # 跳过开高低收价格无效的记录
    allow_missing_years: bool = False  # 是否允许部分年份的数据文件缺失
    max_inflight: int = 5_000       # 同时在处理中的最大任务数
    progress_every: int = 10_000    # 每处理多少条记录报告一次进度
    batch_size: int = 100          # 同一时间点每批处理的股票数量

@dataclass(frozen=True)
class RuntimeConfig:
    """解析后的运行配置，同时保留原始 JSON 数据。"""

    mode: RunMode
    parquet: ParquetConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "config", repr=False)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """从 JSON 文件加载运行配置，并校验模式及 Parquet 参数。

    Args:
        path: 配置文件路径，支持 ``~`` 和相对路径。

    Returns:
        校验后的运行配置。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 缺少运行模式、模式不支持，或 Parquet 配置无效。
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("runtime config must be an object")
    from vnpy.alpha.mining.config import runtime_iteration_config
    runtime_iteration_config(raw.get("expression_iteration", {}), config_path.parent)
    from vnpy.alpha.modeling.evaluation_config import parse_evaluation_config
    parse_evaluation_config(raw.get("factor_evaluation", {}), config_path.parent)
    from vnpy.factor.factor_history_batch import parse_history_batch_config
    batch = parse_history_batch_config(raw.get("history_batch", {}), config_path.parent)
    if batch is not None and raw.get("parquet_import", {}).get("enabled", False):
        raise ValueError("history_batch and parquet_import cannot both be enabled")

    try:
        mode = RunMode(raw["mode"])
    except KeyError as exc:
        raise ValueError("runtime config is missing 'mode'") from exc
    except ValueError as exc:
        supported = ", ".join(item.value for item in RunMode)
        raise ValueError(f"unsupported runtime mode; expected one of: {supported}") from exc

    section = raw.get(mode.value)
    if not isinstance(section, dict):
        raise ValueError(f"runtime config is missing object section '{mode.value}'")

    try:
        parquet = ParquetConfig(**section)
        if parquet.frequency != "1d":
            raise ValueError("parquet only supports frequency '1d'")
        if parquet.max_inflight <= 0:
            raise ValueError("parquet max_inflight must be greater than zero")
        if parquet.progress_every <= 0:
            raise ValueError("parquet progress_every must be greater than zero")
        if isinstance(parquet.batch_size, bool) or not isinstance(parquet.batch_size, int) or parquet.batch_size <= 0:
            raise ValueError("parquet batch_size must be a positive integer")
        return RuntimeConfig(mode=mode, parquet=parquet, raw=raw, config_dir=config_path.parent)
    except TypeError as exc:
        raise ValueError(f"invalid '{mode.value}' configuration: {exc}") from exc

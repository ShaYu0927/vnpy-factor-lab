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
DEFAULT_RUNTIME_CONFIG = Path(__file__).resolve().parents[2] / "config" / "runtime.json"

EVENT_ML_SIGNAL = "eMlSignal"


class RunMode(str, Enum):
    PARQUET = "parquet"


@dataclass(frozen=True)
class ParquetConfig:
    root: str
    start: str
    end: str
    symbols: str | None = None
    markets: str = "SHSE,SZSE"
    frequency: str = "1d"
    skip_zero_volume: bool = True
    skip_invalid_ohlc: bool = True
    allow_missing_years: bool = False
    max_inflight: int = 5_000
    progress_every: int = 10_000


@dataclass(frozen=True)
class RuntimeConfig:
    mode: RunMode
    parquet: ParquetConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

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
        return RuntimeConfig(mode=mode, parquet=parquet, raw=raw)
    except TypeError as exc:
        raise ValueError(f"invalid '{mode.value}' configuration: {exc}") from exc

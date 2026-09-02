from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Any

DEFAULT_FREQUENCY = "60s"
DEFAULT_SUBSCRIPTION_QUERY_DATE = os.getenv("VNPY_SUBSCRIPTION_QUERY_DATE")
DEFAULT_SUBSCRIPTION_FALLBACK_DAYS = int(os.getenv("VNPY_SUBSCRIPTION_FALLBACK_DAYS", "7"))
DEFAULT_FACTOR_MODE = os.getenv("VNPY_FACTOR_MODE", "thread")
DEFAULT_FACTOR_MAX_WORKERS = int(os.getenv("VNPY_FACTOR_MAX_WORKERS", "4"))

DEFAULT_STRATEGY_ID = "a2c12b21-3191-11f1-9539-fa89d2391347"
DEFAULT_GM_TOKEN = "ad3b5bc0baaf82a4572f36cff8242f448063e439"

DEFAULT_BACKTEST_START_TIME = "2026-03-01 08:00:00"
DEFAULT_BACKTEST_END_TIME = "2026-04-30 16:00:00"

DEFAULT_INITIAL_CASH = 10_000_000
DEFAULT_COMMISSION_RATIO = 0.0001
DEFAULT_SLIPPAGE_RATIO = 0.0001
DEFAULT_RUNTIME_CONFIG = Path(__file__).resolve().parents[2] / "config" / "runtime.json"

EVENT_ML_SIGNAL = "eMlSignal"


class RunMode(str, Enum):
    GM_LOCAL = "gm_local"
    GM_SQLITE = "gm_sqlite"
    DATABASE = "database"
    GM_BACKTEST = "gm_backtest"


@dataclass(frozen=True)
class GmLocalConfig:
    symbols: str
    frequency: str = "60s"
    start: str | None = None
    end: str | None = None
    count: int | None = None


@dataclass(frozen=True)
class GmSqliteConfig:
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
class DatabaseConfig:
    path: str
    frequency: str = "60s"
    symbols: str | None = None
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class GmBacktestConfig:
    strategy_id: str
    start: str
    end: str
    initial_cash: float = 10_000_000
    commission_ratio: float = 0.0001
    slippage_ratio: float = 0.0001


@dataclass(frozen=True)
class RuntimeConfig:
    mode: RunMode
    gm_local: GmLocalConfig | None = None
    gm_sqlite: GmSqliteConfig | None = None
    database: DatabaseConfig | None = None
    gm_backtest: GmBacktestConfig | None = None
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
        if mode == RunMode.GM_LOCAL:
            gm_local = GmLocalConfig(**section)
            if gm_local.count is None and (not gm_local.start or not gm_local.end):
                raise ValueError("gm_local requires count, or both start and end")
            return RuntimeConfig(mode=mode, gm_local=gm_local, raw=raw)

        if mode == RunMode.GM_SQLITE:
            gm_sqlite = GmSqliteConfig(**section)
            if gm_sqlite.frequency != "1d":
                raise ValueError("gm_sqlite only supports frequency '1d'")
            if gm_sqlite.max_inflight <= 0:
                raise ValueError("gm_sqlite max_inflight must be greater than zero")
            if gm_sqlite.progress_every <= 0:
                raise ValueError("gm_sqlite progress_every must be greater than zero")
            return RuntimeConfig(mode=mode, gm_sqlite=gm_sqlite, raw=raw)

        if mode == RunMode.DATABASE:
            return RuntimeConfig(
                mode=mode,
                database=DatabaseConfig(**section),
                raw=raw,
            )

        return RuntimeConfig(
            mode=mode,
            gm_backtest=GmBacktestConfig(**section),
            raw=raw,
        )
    except TypeError as exc:
        raise ValueError(f"invalid '{mode.value}' configuration: {exc}") from exc

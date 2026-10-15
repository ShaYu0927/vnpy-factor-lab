from typing import Any

from .alpha_logger import logger


__all__ = [
    "logger",
    "Alpha",
    "MarketFrame",
    "FactorFrame",
    "SessionSplit",
    "TimeSplitConfig",
    "AlphaDefinition",
    "AlphaEngine",
    "AlphaSample",
    "AlphaSampleCache",
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "AlphaModel",
    "BaseModel",
    "FineTunableAlphaModel",
    "Reweighter",
    "ColumnReweighter",
    "AlphaStrategy",
    "BacktestingEngine",
    "AlphaLab",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional alpha components on first access."""
    if name == "Alpha":
        from .alpha_formula import Alpha

        values = {"Alpha": Alpha}
    elif name in {"MarketFrame", "FactorFrame", "SessionSplit", "TimeSplitConfig"}:
        from .alpha_domain import FactorFrame, MarketFrame, SessionSplit, TimeSplitConfig

        values = {
            "MarketFrame": MarketFrame,
            "FactorFrame": FactorFrame,
            "SessionSplit": SessionSplit,
            "TimeSplitConfig": TimeSplitConfig,
        }
    elif name in {"AlphaDefinition", "AlphaEngine", "AlphaSample", "AlphaSampleCache"}:
        from .alpha_definition import AlphaDefinition
        from .alpha_engine import AlphaEngine, AlphaSample, AlphaSampleCache

        values = {
            "AlphaDefinition": AlphaDefinition,
            "AlphaEngine": AlphaEngine,
            "AlphaSample": AlphaSample,
            "AlphaSampleCache": AlphaSampleCache,
        }
    elif name in {"AlphaDataset", "Segment", "to_datetime"}:
        from .dataset import AlphaDataset, Segment, to_datetime

        values = {
            "AlphaDataset": AlphaDataset,
            "Segment": Segment,
            "to_datetime": to_datetime,
        }
    elif name in {
        "AlphaModel",
        "BaseModel",
        "FineTunableAlphaModel",
        "Reweighter",
        "ColumnReweighter",
    }:
        from .model import (
            AlphaModel,
            BaseModel,
            ColumnReweighter,
            FineTunableAlphaModel,
            Reweighter,
        )

        values = {
            "AlphaModel": AlphaModel,
            "BaseModel": BaseModel,
            "FineTunableAlphaModel": FineTunableAlphaModel,
            "Reweighter": Reweighter,
            "ColumnReweighter": ColumnReweighter,
        }
    elif name in {"AlphaStrategy", "BacktestingEngine"}:
        from .strategy import AlphaStrategy, BacktestingEngine

        values = {
            "AlphaStrategy": AlphaStrategy,
            "BacktestingEngine": BacktestingEngine,
        }
    elif name == "AlphaLab":
        from .alpha_lab import AlphaLab

        values = {"AlphaLab": AlphaLab}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(values)
    return values[name]

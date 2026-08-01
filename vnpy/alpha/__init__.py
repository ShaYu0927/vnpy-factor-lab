from typing import Any

from .logger import logger


__all__ = [
    "logger",
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "AlphaModel",
    "AlphaStrategy",
    "BacktestingEngine",
    "AlphaLab"
]


def __getattr__(name: str) -> Any:
    """Lazily import optional alpha components on first access."""
    if name in {"AlphaDataset", "Segment", "to_datetime"}:
        from .dataset import AlphaDataset, Segment, to_datetime

        values = {
            "AlphaDataset": AlphaDataset,
            "Segment": Segment,
            "to_datetime": to_datetime,
        }
    elif name == "AlphaModel":
        from .model import AlphaModel

        values = {"AlphaModel": AlphaModel}
    elif name in {"AlphaStrategy", "BacktestingEngine"}:
        from .strategy import AlphaStrategy, BacktestingEngine

        values = {
            "AlphaStrategy": AlphaStrategy,
            "BacktestingEngine": BacktestingEngine,
        }
    elif name == "AlphaLab":
        from .lab import AlphaLab

        values = {"AlphaLab": AlphaLab}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(values)
    return values[name]

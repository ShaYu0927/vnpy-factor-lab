from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BarSource(str, Enum):
    PARQUET = "parquet"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MarketBar:
    """Market bar shared by Parquet readers, caches, and factor calculations."""

    symbol: str
    bob: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float | None = None
    frequency: str = "60s"
    source: str = BarSource.UNKNOWN.value
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip()
        if not self.symbol:
            raise ValueError("bar symbol is required")
        if not isinstance(self.bob, datetime):
            self.bob = parse_bar_datetime(self.bob)
        if not self.frequency:
            self.frequency = "60s"

        self.open = float(self.open)
        self.high = float(self.high)
        self.low = float(self.low)
        self.close = float(self.close)
        self.volume = float(self.volume)
        self.amount = None if self.amount is None else float(self.amount)

    @property
    def datetime(self) -> datetime:
        return self.bob

    @property
    def open_price(self) -> float:
        return self.open

    @property
    def high_price(self) -> float:
        return self.high

    @property
    def low_price(self) -> float:
        return self.low

    @property
    def close_price(self) -> float:
        return self.close

    @property
    def turnover(self) -> float:
        return self.amount or 0.0


def parse_bar_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"unsupported bar datetime: {value!r}")


BarData = MarketBar

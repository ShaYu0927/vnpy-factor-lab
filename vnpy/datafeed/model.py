from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class BarSource(str, Enum):
    GM_LIVE = "gm-live"
    GM_LOCAL = "gm-local"
    GM_SQLITE = "gm-sqlite"
    SQLITE = "sqlite"
    CSV = "csv"
    VNPY = "vnpy"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MarketBar:
    """Canonical bar for realtime subscriptions and historical replay."""

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


def get_bar_value(raw_bar: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(raw_bar, Mapping):
            value = raw_bar.get(name)
        elif hasattr(raw_bar, "index") and name in raw_bar.index:
            value = raw_bar[name]
        else:
            value = getattr(raw_bar, name, None)
        if value is not None:
            return value
    return default


def normalize_bar(
    raw_bar: Any,
    *,
    frequency: str | None = None,
    source: str | BarSource = BarSource.UNKNOWN,
    default_symbol: str | None = None,
) -> MarketBar:
    """Normalize GM, dict, pandas and VeighNa bars at the input boundary."""
    if isinstance(raw_bar, MarketBar):
        if frequency:
            raw_bar.frequency = frequency
        if source != BarSource.UNKNOWN:
            raw_bar.source = source.value if isinstance(source, BarSource) else source
        return raw_bar

    source_value = source.value if isinstance(source, BarSource) else source
    symbol = get_bar_value(raw_bar, "symbol", "vt_symbol", "code", default=default_symbol)
    bob = get_bar_value(raw_bar, "bob", "datetime", "time", "date")
    if symbol is None:
        raise ValueError("bar is missing symbol/vt_symbol/code")
    if bob is None:
        raise ValueError("bar is missing bob/datetime/time/date")

    bar_frequency = frequency or get_bar_value(raw_bar, "frequency", "interval", default="60s")
    if hasattr(bar_frequency, "value"):
        bar_frequency = bar_frequency.value

    return MarketBar(
        symbol=str(symbol),
        bob=parse_bar_datetime(bob),
        open=get_bar_value(raw_bar, "open", "open_price"),
        high=get_bar_value(raw_bar, "high", "high_price"),
        low=get_bar_value(raw_bar, "low", "low_price"),
        close=get_bar_value(raw_bar, "close", "close_price"),
        volume=get_bar_value(raw_bar, "volume", "vol", default=0.0),
        amount=get_bar_value(raw_bar, "amount", "turnover", default=None),
        frequency=str(bar_frequency),
        source=source_value,
    )


BarData = MarketBar

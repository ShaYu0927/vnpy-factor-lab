
from ast import Dict
from collections import defaultdict, deque
from dataclasses import dataclass
import datetime
from typing import List, Optional, Tuple


    # In-memory cache for gm.api bar data.

    # This cache stores the latest N bars for each symbol and frequency.
    # It is mainly used to provide recent historical bars for factor calculation,
    # signal generation, and strategy decision-making.

    # The cache is organized by:
    #     key:   (symbol, frequency)
    #     value: deque[BarData]

    # Example:
    #     ("SHSE.600000", "60s") -> latest 1000 60-second bars
    #     ("SZSE.000001", "1d")  -> latest 1000 daily bars

@dataclass
class BarData:
    symbol: str                         # 股票
    bob: datetime                       # Bar 开始时间，bob = beginning of bar，例如 09:31:00 这一根K线的开始时间

    open: float                         # 开盘价
    high: float                         # 最高价
    low: float                          # 最低价
    close: float                        # 收盘价

    volume: float                       # 成交量
    amount: Optional[float] = None      # 成交额，可选字段；部分数据源可能没有成交额

    frequency: str = "60s"              # K线周期，默认 60s；例如：60s、300s、900s、1d
    
class BarCache:
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen

        # key: (symbol, frequency)
        # value: deque[BarData]
        self._bars = defaultdict(
            lambda: deque(maxlen=self.maxlen)
        )

    def _make_key(self, symbol: str, frequency: str) -> Tuple[str, str]:
        return symbol, frequency

    def update(self, bar: BarData) -> None:
        if bar is None:
            return

        if not bar.symbol:
            return

        if not bar.frequency:
            bar.frequency = "60s"

        key = self._make_key(bar.symbol, bar.frequency)
        bars = self._bars[key]

        # 如果是同一根K线，就更新最后一根
        if bars and bars[-1].bob == bar.bob:
            bars[-1] = bar
        else:
            bars.append(bar)

    def get_bars(self,symbol: str, count: Optional[int] = None, frequency: str = "60s") -> List[BarData]:
        key = self._make_key(symbol, frequency)
        bars = self._bars.get(key)
        if not bars:
            return []
        bars_list = list(bars)
        if count is None:
            return bars_list
        return bars_list[-count:]

    def get_last_bar(self, symbol: str, frequency: str = "60s") -> Optional[BarData]:
        key = self._make_key(symbol, frequency)
        bars = self._bars.get(key)
        if not bars:
            return None
        return bars[-1]

    def size(self, symbol: str,frequency: str = "60s") -> int:
        key = self._make_key(symbol, frequency)
        bars = self._bars.get(key)

        return len(bars) if bars else 0

    def keys(self):
        return list(self._bars.keys())

    def symbols(self) -> List[str]:
        return list(set(symbol for symbol, _ in self._bars.keys()))

    def clear_symbol(self, symbol: str, frequency: Optional[str] = None) -> None:
        if frequency is not None:
            key = self._make_key(symbol, frequency)
            self._bars.pop(key, None)
            return

        remove_keys = [
            key for key in self._bars.keys()
            if key[0] == symbol
        ]

        for key in remove_keys:
            self._bars.pop(key, None)

    def clear_all(self) -> None:
        self._bars.clear()

def get_bar_value(raw_bar, key: str):
    if isinstance(raw_bar, dict):
        return raw_bar.get(key)

    return getattr(raw_bar, key, None)


def to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def convert_gm_bar(raw_bar, frequency: str = "60s") -> BarData:
    return BarData(
        symbol=get_bar_value(raw_bar, "symbol"),
        bob=get_bar_value(raw_bar, "bob"),
        open=to_float(get_bar_value(raw_bar, "open")),
        high=to_float(get_bar_value(raw_bar, "high")),
        low=to_float(get_bar_value(raw_bar, "low")),
        close=to_float(get_bar_value(raw_bar, "close")),
        volume=to_float(get_bar_value(raw_bar, "volume")),
        amount=to_float(get_bar_value(raw_bar, "amount"), default=0.0),
        frequency=frequency,
    )
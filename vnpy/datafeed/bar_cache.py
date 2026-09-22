"""Recent bars indexed by symbol and frequency for factor calculations."""
from collections import defaultdict, deque

from vnpy.datafeed.model import BarData


class BarCache:
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self._bars: dict[tuple[str, str], deque[BarData]] = defaultdict(
            lambda: deque(maxlen=self.maxlen)
        )

    def update(self, bar: BarData) -> None:
        if bar is None or not bar.symbol:
            return
        if not bar.frequency:
            bar.frequency = "60s"
        bars = self._bars[(bar.symbol, bar.frequency)]
        if bars and bars[-1].bob == bar.bob:
            bars[-1] = bar
        else:
            bars.append(bar)

    def get_bars(self, symbol: str, count: int | None = None, frequency: str = "60s") -> list[BarData]:
        bars = list(self._bars.get((symbol, frequency), ()))
        return bars if count is None else bars[-count:]

    def get_last_bar(self, symbol: str, frequency: str = "60s") -> BarData | None:
        bars = self._bars.get((symbol, frequency))
        return bars[-1] if bars else None

    def size(self, symbol: str, frequency: str = "60s") -> int:
        return len(self._bars.get((symbol, frequency), ()))

    def keys(self) -> list[tuple[str, str]]:
        return list(self._bars)

    def symbols(self) -> list[str]:
        return list({symbol for symbol, _ in self._bars})

    def clear_symbol(self, symbol: str, frequency: str | None = None) -> None:
        if frequency is not None:
            self._bars.pop((symbol, frequency), None)
        else:
            for key in [key for key in self._bars if key[0] == symbol]:
                del self._bars[key]

    def clear_all(self) -> None:
        self._bars.clear()

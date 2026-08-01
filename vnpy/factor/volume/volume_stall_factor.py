from dataclasses import dataclass
from typing import Any

from vnpy.factor.core import Factor, FactorContext


@dataclass(frozen=True)
class VolumeStallResult:
    """
    放量滞涨因子结果

    stall_score 越高，表示放量但价格收不住，短线冲高回落风险越大。
    """
    stall_score: float
    volume_ratio: float
    price_position: float
    upper_shadow_ratio: float
    recent_high_gap: float
    window: int
    volume_window: int

class VolumeStallFactor:
    def __init__(self, window: int = 20, price_threshold: float = 0.02, volume_threshold: float = 0.20, divergence_threshold: float = 0.20,) -> None:
            self.window = window
            self.divergence_threshold = divergence_threshold
            self.price_threshold = price_threshold
            self.volume_threshold = volume_threshold

    def calculate(self, symbol: str, opens: list[float], highs: list[float], lows: list[float], closes: list[float], volumes: list[float],) -> VolumeStallResult:
        min_len = self.window
        if len(opens) < min_len or len(highs) < min_len or len(lows) < min_len:
            raise ValueError(f"{symbol} bars length must be >= {min_len}")

        open_price = opens[-1]
        high_price = highs[-1]
        low_price = lows[-1]
        close_price = closes[-1]

        price_range = high_price - low_price
        if price_range <= 0:
            price_position = 0.5
            upper_shadow_ratio = 0.0
        else:
            price_position = (close_price - low_price) / price_range
            upper_shadow_ratio = (high_price - max(open_price, close_price)) / price_range

        recent_volumes = volumes[-self.window:]
        base_volumes = volumes[-self.volume_window:]

        recent_volume_avg = sum(recent_volumes) / len(recent_volumes)
        base_volume_avg = sum(base_volumes) / len(base_volumes)

        if base_volume_avg <= 0:
            volume_ratio = 0.0
        else:
            volume_ratio = recent_volume_avg / base_volume_avg

        stall_score = volume_ratio * (0.7 * (1.0 - price_position) + 0.3 * upper_shadow_ratio)

        is_volume_stall = (volume_ratio >= self.volume_ratio_threshold and stall_score >= self.stall_threshold)

        return VolumeStallResult(
            symbol=symbol,
            stall_score=stall_score,
            volume_ratio=volume_ratio,
            price_position=price_position,
            upper_shadow_ratio=upper_shadow_ratio,
            is_volume_stall=is_volume_stall,
            window=self.window,
            volume_window=self.volume_window,
        )


def _get_value(bar: Any, field: str) -> Any:
    if isinstance(bar, dict):
        return bar.get(field)
    return getattr(bar, field, None)


class VolumeStallEngineFactor(Factor):
        def __init__(self, window: int = 20, price_threshold: float = 0.02, volume_threshold: float = 0.20, divergence_threshold: float = 0.20,) -> None:
            self.window = window
            self.price_threshold = price_threshold
            self.volume_threshold = volume_threshold
            self.divergence_threshold = divergence_threshold
            self.factor = VolumeStallFactor(window, price_threshold, volume_threshold, divergence_threshold)

        def _extract_values(self, data: Any, field: str) -> list[float]:
            values: list[float] = []

            for bar in data:
                value = self._get_value(bar, field)
                if value is None:
                    continue
                values.append(float(value))

            return values

        def calculate(self, symbol: str, data: Any, context: FactorContext, ) -> VolumeStallResult:
            closes = self._extract_values(data, "close")
            volumes = self._extract_values(data, "volume")
            opens = self._extract_values(data, "open")
            highs = self._extract_values(data, "high")
            lows = self._extract_values(data, "low")

            return self.factor.calculate(symbol=symbol, opens = opens, highs = highs, lows = lows, closes = closes, volumes = volumes)
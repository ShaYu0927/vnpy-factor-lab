from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from vnpy.factor.core.factorEngine import Factor, FactorContext


@dataclass
class VolumePriceDivergenceResult:
    """
    Result of volume-price divergence calculation.
    """

    symbol: str
    window: int
    price_trend: float = 0.0
    volume_trend: float = 0.0
    corr: float = 0.0
    divergence: float = 0.0
    score: float = 0.0
    price_return: float = 0.0
    volume_change: float = 0.0
    signal: str = "UNKNOWN"
    reason: str = ""


class VolumePriceDivergenceFactor:
    """
    Volume-price divergence factor.

    Formula:
        factor = -1 * corr(close / lag(close, 1), log(volume / lag(volume, 1) + 1), window)

    Meaning:
        corr > 0:
            Price change and volume change move in the same direction.
            This is usually trend confirmation.

        corr < 0:
            Price change and volume change move in opposite directions.
            This is volume-price divergence.

        score = -corr:
            Higher score means stronger divergence.
    """

    def __init__(self, window: int = 20, price_threshold: float = 0.02, volume_threshold: float = 0.20, divergence_threshold: float = 0.20) -> None:
        self.window = window
        self.divergence_threshold = divergence_threshold
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold

    def calculate(self, symbol: str, closes: list[float], volumes: list[float],) -> VolumePriceDivergenceResult:
        result = VolumePriceDivergenceResult(symbol=symbol, window=self.window)

        if len(closes) < self.window + 1 or len(volumes) < self.window + 1:
            result.reason = (f"Insufficient data: at least {self.window + 1} closes and volumes are required")
            return result
        
        pairs = self._clean_tail_pairs(closes, volumes, self.window + 1)

        if len(pairs) < self.window + 1:
            result.reason = "Invalid data: close or volume contains missing values"
            return result

        recent_closes = [item[0] for item in pairs]
        recent_volumes = [item[1] for item in pairs]

        price_ratios: list[float] = []
        volume_log_ratios: list[float] = []

        for i in range(1, len(pairs)):
            prev_close = recent_closes[i - 1]
            curr_close = recent_closes[i]

            prev_volume = recent_volumes[i - 1]
            curr_volume = recent_volumes[i]

            if prev_close <= 0 or curr_close <= 0:
                result.reason = "Invalid data: close must be positive"
                return result

            if prev_volume <= 0 or curr_volume < 0:
                result.reason = "Invalid data: volume must be non-negative and previous volume must be positive"
                return result

            price_ratio = curr_close / prev_close
            volume_log_ratio = math.log(curr_volume / prev_volume + 1.0)

            price_ratios.append(price_ratio)
            volume_log_ratios.append(volume_log_ratio)

        corr = self._corr(price_ratios, volume_log_ratios)

        result.corr = corr
        result.divergence = -corr

        result.price_return = recent_closes[-1] / recent_closes[0] - 1.0
        result.volume_change = recent_volumes[-1] / recent_volumes[0] - 1.0
        result.price_trend = result.price_return
        result.volume_trend = result.volume_change

        result.signal = self._classify(result.price_trend, result.volume_trend)
        result.score = self._score(result.price_trend, result.volume_trend, result.signal)
        result.reason = self._build_reason(result)

        return result


    def _classify(self, price_trend: float, volume_trend: float) -> str:
        price_down = price_trend <= -self.price_threshold
        price_up = price_trend >= self.price_threshold
        volume_down = volume_trend <= -self.volume_threshold
        volume_up = volume_trend >= self.volume_threshold

        if price_down and volume_up:
            return "BULLISH_DIVERGENCE"

        if price_up and volume_down:
            return "BEARISH_DIVERGENCE"

        if price_down and volume_down:
            return "SELLING_EXHAUSTION"

        if price_up and volume_up:
            return "TREND_CONFIRMATION"

        return "NEUTRAL"

    def _score(self, price_trend: float, volume_trend: float, signal: str) -> float:
        if signal == "BULLISH_DIVERGENCE":
            return abs(price_trend) + volume_trend

        if signal == "SELLING_EXHAUSTION":
            return abs(price_trend) * 0.5 + abs(volume_trend) * 0.5

        if signal == "TREND_CONFIRMATION":
            return price_trend + volume_trend * 0.25

        if signal == "BEARISH_DIVERGENCE":
            return -(price_trend + abs(volume_trend))

        return 0.0

    def _normalized_slope(self, values: list[float]) -> float:
        base = values[0]
        normalized = [value / base - 1.0 for value in values]

        n = len(normalized)
        x_mean = (n - 1) / 2
        y_mean = sum(normalized) / n

        numerator = 0.0
        denominator = 0.0
        for index, value in enumerate(normalized):
            x_delta = index - x_mean
            numerator += x_delta * (value - y_mean)
            denominator += x_delta * x_delta

        if denominator <= 0:
            return 0.0

        return numerator / denominator * (n - 1)

    def _clean_tail(self, values: list[float], count: int) -> list[float]:
        result: list[float] = []

        for value in values[-count:]:
            if value is None:
                continue
            result.append(float(value))

        return result
    
    def _clean_tail_pairs(self, closes: list[float], volumes: list[float], count: int,) -> list[tuple[float, float]]:
        """
        Extract the latest close-volume pairs.

        This function keeps the mapping relationship between close and volume:

            closes[i] <-> volumes[i]

        That means each returned pair represents data from the same bar / same trading day:

            (close_price_of_bar_i, volume_of_bar_i)

        This mapping is important for volume-price divergence calculation, because the
        factor compares daily price change and daily volume change:

            price_ratio[i] = close[i] / close[i - 1]
            volume_ratio[i] = volume[i] / volume[i - 1]

        Therefore, price_ratio[i] and volume_ratio[i] must come from the same time point.
        If close and volume are filtered separately, the two sequences may become
        misaligned and the correlation result will be incorrect.

        Args:
            closes: Close price sequence.
            volumes: Volume sequence.
            count: Number of latest records to keep.

        Returns:
            A list of valid (close, volume) pairs.
        """
        
        pairs: list[tuple[float, float]] = []

        tail_closes = closes[-count:]
        tail_volumes = volumes[-count:]

        for close, volume in zip(tail_closes, tail_volumes):
            if close is None or volume is None:
                continue

            pairs.append((float(close), float(volume)))

        return pairs
    
    def _corr(self, xs: list[float], ys: list[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            return 0.0

        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        numerator = 0.0
        x_var = 0.0
        y_var = 0.0

        for x, y in zip(xs, ys):
            x_delta = x - x_mean
            y_delta = y - y_mean

            numerator += x_delta * y_delta
            x_var += x_delta * x_delta
            y_var += y_delta * y_delta

        denominator = math.sqrt(x_var * y_var)

        if denominator <= 0:
            return 0.0

        return numerator / denominator

    def _build_reason(self, result: VolumePriceDivergenceResult) -> str:
        return (
            f"price_trend={result.price_trend:.6f}, "
            f"volume_trend={result.volume_trend:.6f}, "
            f"corr={result.corr:.6f}, "
            f"divergence={result.divergence:.6f}, "
            f"signal={result.signal}, "
            f"score={result.score:.6f}"
        )


class VolumePriceDivergenceEngineFactor(Factor):
    """
    FactorEngine adapter for VolumePriceDivergenceFactor.
    """

    def __init__(
        self,
        window: int = 20,
        price_threshold: float = 0.02,
        volume_threshold: float = 0.20,
        divergence_threshold: float = 0.20,
    ) -> None:
        self.window = window
        self.name = f"volume_price_divergence_{window}"
        self.min_bars = window + 1
        self.factor = VolumePriceDivergenceFactor(
            window=window,
            price_threshold=price_threshold,
            volume_threshold=volume_threshold,
            divergence_threshold=divergence_threshold,
        )

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        return data is not None and len(data) >= self.min_bars

    def calculate(self, symbol: str, data: Any, context: FactorContext,) -> VolumePriceDivergenceResult:
        closes = self._extract_values(data, "close")
        volumes = self._extract_values(data, "volume")

        return self.factor.calculate(symbol=symbol, closes=closes, volumes=volumes,)

    def _extract_values(self, data: Any, field: str) -> list[float]:
        values: list[float] = []

        for bar in data:
            value = self._get_value(bar, field)
            if value is None:
                continue
            values.append(float(value))

        return values

    def _get_value(self, bar: Any, field: str) -> Any:
        if isinstance(bar, dict):
            return bar.get(field)

        return getattr(bar, field, None)

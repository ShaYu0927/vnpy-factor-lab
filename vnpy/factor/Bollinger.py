from dataclasses import dataclass
import math
from typing import List


@dataclass
class BollingerResult:
    """
    Bollinger Bands factor result for a single stock.
    """

    symbol: str
    window: int
    # Latest close price.
    close: float = 0.0
    # Middle band: moving average of close prices.
    middle: float = 0.0
    # Upper band: middle + num_std * standard deviation.
    upper: float = 0.0
    # Lower band: middle - num_std * standard deviation.
    lower: float = 0.0
    # Standard deviation of close prices.
    std: float = 0.0
    # Band width: (upper - lower) / middle.
    band_width: float = 0.0
    # Position of current close inside Bollinger Bands.
    # < 0: below lower band
    # 0 ~ 1: inside bands
    # > 1: above upper band
    position: float = 0.0
    # Signal label.
    # BREAK_UP / BREAK_DOWN / NEAR_UPPER / NEAR_LOWER / MIDDLE / UNKNOWN
    signal: str = "UNKNOWN"
    # Explanation.
    reason: str = ""
    
    
class BollingerFactor:
    """
    Bollinger Bands factor.

    Core idea:
        Bollinger Bands use moving average and standard deviation
        to describe the price fluctuation channel.

    Formula:
        middle = mean(close)
        upper = middle + num_std * std(close)
        lower = middle - num_std * std(close)

    Common usage:
        1. Identify whether price is close to upper or lower band.
        2. Detect breakout above upper band or below lower band.
        3. Measure volatility expansion or contraction using band width.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0,near_threshold: float = 0.8,):
        self.window = window
        self.num_std = num_std
        self.near_threshold = near_threshold
        
    def calculate(self, symbol: str, closes: List[float]) -> BollingerResult:
        """
        Calculate Bollinger Bands factor for a single stock.

        Args:
            symbol:
                Stock symbol.

            closes:
                Close price sequence.
                At least window close prices are required.

        Returns:
            BollingerResult.
        """
        result = BollingerResult(symbol=symbol, window=self.window)
        if len(closes) < self.window:
            result.reason = (f"Insufficient data: at least {self.window} close prices are required")
            return result
        
        recent_closes = closes[-self.window:]
        close_now = recent_closes[-1]
        
        result.close = close_now
        
        # Middle band: moving average.
        result.middle = sum(recent_closes) / len(recent_closes)
        # Standard deviation of close prices.
        result.std = self._calc_std(recent_closes)

        # Upper and lower bands.
        result.upper = result.middle + self.num_std * result.std
        result.lower = result.middle - self.num_std * result.std
        
        # Band width.
        if result.middle > 0:
            result.band_width = (result.upper - result.lower) / result.middle
        else:
            result.band_width = 0.0
            
        # Position inside Bollinger Bands.
        band_range = result.upper - result.lower
        if band_range > 0:
            result.position = (close_now - result.lower) / band_range
        else:
            result.position = 0.5
            
        # Signal classification.
        if close_now > result.upper:
            result.signal = "BREAK_UP"
            result.reason = (f"Close price {close_now:.4f} is above the upper band " f"{result.upper:.4f}. Price breaks above Bollinger Bands.")
        elif close_now < result.lower:
            result.signal = "BREAK_DOWN"
            result.reason = (f"Close price {close_now:.4f} is below the lower band " f"{result.lower:.4f}. Price breaks below Bollinger Bands.")
        elif result.position >= self.near_threshold:
            result.signal = "NEAR_UPPER"
            result.reason = (f"Close price is near the upper band. " f"Position={result.position:.4f}.")
        elif result.position <= 1.0 - self.near_threshold:
            result.signal = "NEAR_LOWER"
            result.reason = (f"Close price is near the lower band. " f"Position={result.position:.4f}.")
        else:
            result.signal = "MIDDLE"
            result.reason = (f"Close price is inside the middle area of Bollinger Bands. " f"Position={result.position:.4f}.")

        return result



    def _calc_std(self, values: List[float]) -> float:
        """
        Calculate population standard deviation.
        """
        if not values:
            return 0.0

        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)

        return math.sqrt(variance)

from dataclasses import dataclass
import math
from typing import List


@dataclass
class MomentumResult:
    """
    Momentum factor result for a single stock.
    """
    symbol: str
    window: int

    # Latest one-bar return.
    ret_1: float = 0.0
    # Return over the latest window bars.
    ret_n: float = 0.0
    # Core momentum factor value.
    momentum: float = 0.0
    # Momentum strength adjusted by price range.
    momentum_strength: float = 0.0
    # Trend label: UP / DOWN / FLAT.
    trend: str = "UNKNOWN"
    # Explanation.
    reason: str = ""


class MomentumFactor:
    """
    Momentum factor.

    Core idea:
        Measure how much the latest close price has changed
        compared with the close price N bars ago.

    Common usage:
        1. Measure short-term strength.
        2. Rank stocks by recent performance.
        3. Generate auxiliary trading signals.
    """

    def __init__(self, window: int = 5, up_threshold: float = 0.0, down_threshold: float = 0.0):
        """
        Args:
            window:
                Momentum lookback window.
                For example, window=5 means calculating the return
                over the latest 5 bars.

            up_threshold:
                Threshold for judging an upward trend.
                Example: 0.02 means the stock is considered UP
                only if the latest N-bar return is above 2%.

            down_threshold:
                Threshold for judging a downward trend.
                Example: -0.02 means the stock is considered DOWN
                if the latest N-bar return is below -2%.
        """
        self.window = window
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def calculate(self, symbol: str, closes: List[float]) -> MomentumResult:
        """
        Calculate the momentum factor for a single stock.

        Args:
            symbol:
                Stock symbol.

            closes:
                Close price sequence.
                At least window + 1 close prices are required.

        Returns:
            MomentumResult.
        """
        result = MomentumResult(symbol=symbol, window=self.window)

        if len(closes) < self.window + 1:
            result.reason = (f"Insufficient data: at least {self.window + 1} close prices are required")
            return result

        close_now = closes[-1]
        close_prev = closes[-2]
        close_n = closes[-(self.window + 1)]

        if close_prev <= 0 or close_n <= 0:
            result.reason = "Invalid close price: previous close or lookback close is not positive"
            return result

        # Latest one-bar return.
        result.ret_1 = close_now / close_prev - 1

        # N-bar return.
        result.ret_n = close_now / close_n - 1

        # Core momentum value.
        result.momentum = result.ret_n

        # Use price range to normalize momentum strength.
        recent_closes = closes[-(self.window + 1):]
        price_range = max(recent_closes) - min(recent_closes)

        if price_range > 0:
            result.momentum_strength = result.ret_n / price_range
        else:
            result.momentum_strength = 0.0

        # Trend classification.
        if result.ret_n >= self.up_threshold:
            result.trend = "UP"
            result.reason = (
                f"The latest {self.window}-bar return is {result.ret_n:.6f}, "
                f"which is above the up threshold {self.up_threshold:.6f}. "
                f"Short-term momentum is positive."
            )
        elif result.ret_n <= self.down_threshold:
            result.trend = "DOWN"
            result.reason = (
                f"The latest {self.window}-bar return is {result.ret_n:.6f}, "
                f"which is below the down threshold {self.down_threshold:.6f}. "
                f"Short-term momentum is negative."
            )
        else:
            result.trend = "FLAT"
            result.reason = (
                f"The latest {self.window}-bar return is {result.ret_n:.6f}, "
                f"which is between {self.down_threshold:.6f} and {self.up_threshold:.6f}. "
                f"Short-term momentum is flat."
            )
        return result
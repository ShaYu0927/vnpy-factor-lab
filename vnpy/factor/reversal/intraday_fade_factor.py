from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vnpy.factor.core.factorEngine import Factor, FactorContext


@dataclass
class IntradayFadeResult:
    """盘中冲高回落因子结果"""
    symbol: str          # 股票代码
    factor: float        # 因子值，越大表示冲高回落越明显
    rise: float          # 冲高幅度：(high - open) / open
    fall_back: float     # 高位回落幅度：(high - close/current) / high
    fall_ratio: float    # 回落占比：(high - close/current) / (high - low)
    volume_ratio: float  # 放量倍数：window_volume / avg_volume
    signal: bool         # 是否触发冲高回落信号
    
class IntradayFadeSignalConfirm:
    """
    冲高回落实时信号确认器
    用于过滤实时因子跳动，要求连续多次触发后才确认信号
    """

    def __init__(self, confirm_count: int = 3):
        self.confirm_count = confirm_count
        self.signal_count = {}

    def update(self, result: IntradayFadeResult) -> bool:
        if result.signal:
            self.signal_count[result.symbol] = self.signal_count.get(result.symbol, 0) + 1
        else:
            self.signal_count[result.symbol] = 0

        return self.signal_count[result.symbol] >= self.confirm_count
    
class IntradayFadeFactor:
    """
    冲高回落强度因子：
    用于识别短线拉升失败、放量回落、承接偏弱的股票
    """
    def __init__(
        self,
        volume_window: int = 20,
        rise_threshold: float = 0.02,
        fall_back_threshold: float = 0.01,
        fall_ratio_threshold: float = 0.6,
        volume_ratio_threshold: float = 1.5,
    ):
        self.volume_window = volume_window
        self.rise_threshold = rise_threshold
        self.fall_back_threshold = fall_back_threshold
        self.fall_ratio_threshold = fall_ratio_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        
    def calculate(self, symbol: str, open_price: float, high_price: float, low_price: float, current_price: float, current_volume: float, avg_volume: float,) -> Optional[IntradayFadeResult]:
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or current_price <= 0:
            return None
        
        if high_price <= low_price:
            return None
        
        if avg_volume <= 0:
            return None
        
        rise = (high_price - open_price) / open_price
        fall_back = (high_price - current_price) / high_price
        fall_ratio = (high_price - current_price) / (high_price - low_price)
        volume_ratio = current_volume / avg_volume

        factor = rise * fall_back * fall_ratio * volume_ratio
        signal = (
            rise >= self.rise_threshold
            and fall_back >= self.fall_back_threshold
            and fall_ratio >= self.fall_ratio_threshold
            and volume_ratio >= self.volume_ratio_threshold
        )

        return IntradayFadeResult(
            symbol=symbol,
            factor=factor,
            rise=rise,
            fall_back=fall_back,
            fall_ratio=fall_ratio,
            volume_ratio=volume_ratio,
            signal=signal,
        )


class IntradayFadeEngineFactor(Factor):
    """
    FactorEngine adapter for IntradayFadeFactor.
    """

    def __init__(
        self,
        volume_window: int = 20,
        rise_threshold: float = 0.02,
        fall_back_threshold: float = 0.01,
        fall_ratio_threshold: float = 0.6,
        volume_ratio_threshold: float = 1.5,
    ) -> None:
        self.volume_window = volume_window
        self.name = f"intraday_fade_{volume_window}"
        self.min_bars = volume_window + 1
        self.factor = IntradayFadeFactor(
            volume_window=volume_window,
            rise_threshold=rise_threshold,
            fall_back_threshold=fall_back_threshold,
            fall_ratio_threshold=fall_ratio_threshold,
            volume_ratio_threshold=volume_ratio_threshold,
        )

    def validate(self, symbol: str, data: Any, context: FactorContext) -> bool:
        return data is not None and len(data) >= self.min_bars

    def calculate(self, symbol: str, data: Any, context: FactorContext) -> Optional[IntradayFadeResult]:
        if len(data) < self.min_bars:
            return None

        window_bars = data[-self.volume_window:]
        first_bar = window_bars[0]
        latest_bar = window_bars[-1]

        latest_bar = data[-1]
        avg_volume = self._average_previous_volume(data)

        open_price = self._get_float(first_bar, "open")
        high_price = max(self._get_float(bar, "high") for bar in window_bars)
        low_price = min(self._get_float(bar, "low") for bar in window_bars)
        current_price = self._get_float(latest_bar, "close")
        current_volume = sum(self._get_float(bar, "volume") for bar in window_bars)

        avg_volume = self._average_previous_window_volume(data)
        
        return self.factor.calculate(
            symbol=symbol,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            current_price=current_price,
            current_volume=current_volume,
            avg_volume=avg_volume,
        )

    def _average_previous_volume(self, data: Any) -> float:
        volumes = [
            self._get_float(bar, "volume")
            for bar in data[:-1][-self.volume_window:]
        ]
        volumes = [volume for volume in volumes if volume > 0]

        if not volumes:
            return 0.0

        return sum(volumes) / len(volumes)

    def _get_float(self, bar: Any, field: str) -> float:
        if isinstance(bar, dict):
            value = bar.get(field, 0.0)
        else:
            value = getattr(bar, field, 0.0)

        if value is None:
            return 0.0

        return float(value)

    def _average_previous_window_volume(self, data: Any) -> float:
        if len(data) < self.volume_window * 2:
            return 0.0

        previous_window = data[-self.volume_window * 2:-self.volume_window]

        volumes = [
            self._get_float(bar, "volume")
            for bar in previous_window
        ]

        volumes = [volume for volume in volumes if volume > 0]

        if not volumes:
            return 0.0

        return sum(volumes)
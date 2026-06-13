from typing import Optional
from vnpy.datafeed.BarCache import BarCache
from ..volatility import VolatilityFactor, VolatilityResult
from ..volume import VolumeFactorResult, VolumeFactor
from ..momentum import MomentumResult, MomentumFactor

from dataclasses import dataclass


@dataclass
class BasicFactorConfig:
    """
    Basic factor calculator configuration.
    """

    frequency: str = "60s"

    momentum_window: int = 20
    volatility_window: int = 20
    volume_window: int = 20

    momentum_up_threshold: float = 0.002
    momentum_down_threshold: float = -0.002


@dataclass
class BasicFactorResult:
    """
    All basic factor results of one symbol.
    """

    symbol: str
    momentum: Optional[MomentumResult] = None
    volatility: Optional[VolatilityResult] = None
    volume: Optional[VolumeFactorResult] = None
    
    
class BasicFactorCalculator:
    """
    基础行情因子计算器。

    职责：
    1. 从 BarCache 获取历史 K 线；
    2. 组织基础因子计算；
    3. 对外提供统一的因子计算入口。

    不负责：
    1. 数据订阅
    2. 下单交易
    3. 风控判断
    4. 多线程调度
    """

    def __init__(
        self,
        bar_cache,
        config: Optional[BasicFactorConfig] = None,
    ):
        self.bar_cache = bar_cache
        self.config = config or BasicFactorConfig()

        self.momentum_factor = MomentumFactor(
            window=self.config.momentum_window,
            up_threshold=self.config.momentum_up_threshold,
            down_threshold=self.config.momentum_down_threshold,
        )

        self.volatility_factor = VolatilityFactor(
            window=self.config.volatility_window,
        )

        self.volume_factor = VolumeFactor(
            window=self.config.volume_window,
        )

    def calculate_all(self, symbol: str) -> Optional[BasicFactorResult]:
        """
        计算单只股票的所有基础因子。
        """

        max_window = max(
            self.config.momentum_window,
            self.config.volatility_window,
            self.config.volume_window,
        )

        history_bars = self.bar_cache.get_bars(
            symbol,
            frequency=self.config.frequency,
            count=max_window + 1,
        )

        if len(history_bars) < max_window + 1:
            return None

        closes = [item.close for item in history_bars]
        volumes = [item.volume for item in history_bars]

        momentum_result = self._calculate_momentum(symbol, closes)
        volatility_result = self._calculate_volatility(symbol, closes)
        volume_result = self._calculate_volume(symbol, volumes, closes)

        return BasicFactorResult(
            symbol=symbol,
            momentum=momentum_result,
            volatility=volatility_result,
            volume=volume_result,
        )

    def momentum(self, symbol: str) -> Optional[MomentumResult]:
        """
        计算动量因子。
        """

        history_bars = self._get_history_bars(
            symbol=symbol,
            window=self.config.momentum_window,
        )

        if not history_bars:
            return None

        closes = [item.close for item in history_bars]
        return self._calculate_momentum(symbol, closes)

    def volatility(self, symbol: str) -> Optional[VolatilityResult]:
        """
        计算波动率因子。
        """

        history_bars = self._get_history_bars(
            symbol=symbol,
            window=self.config.volatility_window,
        )

        if not history_bars:
            return None

        closes = [item.close for item in history_bars]
        return self._calculate_volatility(symbol, closes)

    def volume(self, symbol: str) -> Optional[VolumeFactorResult]:
        """
        计算成交量因子。
        """

        history_bars = self._get_history_bars(
            symbol=symbol,
            window=self.config.volume_window,
        )

        if not history_bars:
            return None

        closes = [item.close for item in history_bars]
        volumes = [item.volume for item in history_bars]

        return self._calculate_volume(symbol, volumes, closes)

    def _get_history_bars(self, symbol: str, window: int):
        """
        从 BarCache 获取历史 K 线。
        """

        history_bars = self.bar_cache.get_bars(
            symbol,
            frequency=self.config.frequency,
            count=window + 1,
        )

        if len(history_bars) < window + 1:
            return None

        return history_bars

    def _calculate_momentum(
        self,
        symbol: str,
        closes,
    ) -> Optional[MomentumResult]:
        """
        内部动量因子计算。
        """

        window = self.config.momentum_window

        if len(closes) < window + 1:
            return None

        return self.momentum_factor.calculate(symbol, closes)

    def _calculate_volatility(
        self,
        symbol: str,
        closes,
    ) -> Optional[VolatilityResult]:
        """
        内部波动率因子计算。
        """

        window = self.config.volatility_window

        if len(closes) < window + 1:
            return None

        return self.volatility_factor.calculate(symbol, closes)

    def _calculate_volume(
        self,
        symbol: str,
        volumes,
        closes,
    ) -> Optional[VolumeFactorResult]:
        """
        内部成交量因子计算。
        """

        window = self.config.volume_window

        if len(volumes) < window + 1 or len(closes) < window + 1:
            return None

        return self.volume_factor.calculate(symbol, closes, volumes)
    
    
    

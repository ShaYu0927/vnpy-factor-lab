from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque

from vnpy.strategy.StratContext import (
    SignalDirection,
    StrategyContext,
    StrategySignal,
)
from vnpy.strategy.StratTemplate import StrategyOutput, StrategyTemplate


class BottomTrendInflectionStrategy(StrategyTemplate):
    """
    Bottom trend inflection strategy.

    This strategy receives factor samples and emits LONG/FLAT signals.
    It does not send orders directly.
    """

    author = "strategy"

    def on_init(self, context: StrategyContext) -> None:
        """
        Initialize strategy parameters and runtime state.
        """
        
        self.lookback = int(self.setting.get("lookback", 6))                            # 最近样本窗口长度，用于缓存每只股票最近 N 个因子样本
        self.min_down_bars = int(self.setting.get("min_down_bars", 3))                  # 最近窗口内至少需要出现的下跌 bar 数量

        self.drawdown_threshold = float(self.setting.get("drawdown_threshold", 0.45))   # 最大回撤阈值，0.45 表示从阶段高点下跌 45%
        self.min_volume_ratio = float(self.setting.get("min_volume_ratio", 2.5))        # 最小放量倍数，2.5 表示成交量至少是均量的 2.5 倍
        self.min_return = float(self.setting.get("min_return", 0.08))                   # 最小涨幅阈值，0.08 表示当日涨幅至少 8%
        self.max_distance_from_low = float(self.setting.get("max_distance_from_low", 0.15))  # 距离近期低点的最大涨幅限制，避免追高

        self.stop_loss = float(self.setting.get("stop_loss", 0.025))                    # 止损比例，0.025 表示亏损 2.5% 出场
        self.take_profit = float(self.setting.get("take_profit", 0.05))                 # 止盈比例，0.05 表示盈利 5% 出场

        self.recent_samples: DefaultDict[str, Deque[Any]] = defaultdict(lambda: deque(maxlen=self.lookback))   # 每只股票最近的因子样本缓存

        self.in_position: dict[str, bool] = defaultdict(bool)                           # 每只股票当前是否持仓
        self.entry_price: dict[str, float] = {}                                         # 每只股票的入场价格，用于止盈止损
        self.last_signal_direction: dict[str, SignalDirection] = {}                     # 每只股票最近一次发出的信号方向

        self.write_log(context, "initialized")

    def on_start(self, context: StrategyContext) -> None:
        self.write_log(context, "started")

    def on_stop(self, context: StrategyContext) -> None:
        self.write_log(context, "stopped")

    def on_factor(self, context: StrategyContext, sample: Any, factor_result: Any = None,) -> list[StrategyOutput]:
        """
        Handle factor sample and return strategy signals.
        """

        if sample is None:
            return []

        symbol = getattr(sample, "symbol", "")
        if not symbol:
            return []

        should_enter = self._should_enter(sample)
        should_exit = self._should_exit(sample, symbol)

        self.recent_samples[symbol].append(sample)

        if not self.in_position[symbol] and should_enter:
            self.in_position[symbol] = True
            self.entry_price[symbol] = self._to_float(getattr(sample, "close", 0.0))

            return [
                self._make_signal(
                    sample=sample,
                    direction=SignalDirection.LONG,
                    score=1.0,
                    reason="bottom trend inflection confirmed",
                )
            ]

        if self.in_position[symbol] and should_exit:
            self.in_position[symbol] = False
            self.entry_price.pop(symbol, None)

            return [
                self._make_signal(
                    sample=sample,
                    direction=SignalDirection.FLAT,
                    score=0.0,
                    reason="exit condition triggered",
                )
            ]

        return []
    
    def _shoudler_enter(self, sample: Any) -> bool:
        drawdown_120 = self._to_float(getattr(sample, "drawdown_120", 0.0))

        macd_bottom_divergence = bool(getattr(sample, "macd_bottom_divergence", False))
        return_1d = self._to_float(getattr(sample, "return_1d", 0.0))                    # 当日涨幅
        volume_ratio_10 = self._to_float(getattr(sample, "volume_ratio_10", 0.0))        # 成交量相对前 10 日均量的倍数
        distance_from_low = self._to_float( getattr(sample, "distance_from_low", 999.0)) # 当前价格距离近期最低点的涨幅
        support_hold = bool(getattr(sample, "support_hold_after_surge", False))          # 放量长阳后是否守住支撑
        
        
        return (
            drawdown_120 >= self.drawdown_threshold
            and macd_bottom_divergence
            and return_1d >= self.min_return
            and volume_ratio_10 >= self.min_volume_ratio
            and distance_from_low <= self.max_distance_from_low
            and support_hold
        )

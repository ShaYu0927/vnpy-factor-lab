from __future__ import annotations

from typing import Any

from vnpy.strategy.ml_signal_strategy import MlSignalStrategy, SignalAction
from vnpy.strategy.StratTemplate import StrategyTemplate, StrategyOutput
from vnpy.strategy.StratContext import StrategyContext, StrategySignal, SignalDirection


class FactorSignalStrategy(StrategyTemplate):
    """
    Convert FactorSample into StrategySignal.

    This strategy only generates trading opinions.
    It does not send orders directly.
    """

    def on_init(self, context: StrategyContext) -> None:
        self.signal_model = MlSignalStrategy(
            min_buy_momentum=float(self.setting.get("min_buy_momentum", 0.001)),
            min_sell_momentum=float(self.setting.get("min_sell_momentum", -0.001)),
            max_buy_volatility=float(self.setting.get("max_buy_volatility", 0.03)),
            high_volume_ratio=float(self.setting.get("high_volume_ratio", 1.2)),
            enable_log=bool(self.setting.get("enable_log", True)),
            enable_print=bool(self.setting.get("enable_print", True)),
        )

        self.last_signal = None

    def on_factor(
        self,
        context: StrategyContext,
        sample: Any,
        factor_result: Any = None,
    ) -> list[StrategyOutput]:
        if sample is None:
            return []

        signal = self.signal_model.generate_signal(sample)
        self.last_signal = signal

        if signal is None:
            return []

        symbol = getattr(signal, "symbol", None) or getattr(sample, "symbol", None)

        if not symbol:
            return []

        score = float(getattr(signal, "score", 0.0))
        confidence = float(getattr(signal, "confidence", 0.0))
        reason = getattr(signal, "reason", "")

        if signal.action == SignalAction.BUY:
            return [
                StrategySignal(
                    strategy_name=self.strategy_name,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    score=score,
                    confidence=confidence,
                    reason=reason,
                    extra={
                        "source": "factor_sample",
                        "close": float(getattr(sample, "close", 0.0)),
                    },
                )
            ]

        if signal.action == SignalAction.SELL:
            return [
                StrategySignal(
                    strategy_name=self.strategy_name,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    score=score,
                    confidence=confidence,
                    reason=reason,
                    extra={
                        "source": "factor_sample",
                        "close": float(getattr(sample, "close", 0.0)),
                    },
                )
            ]

        return []

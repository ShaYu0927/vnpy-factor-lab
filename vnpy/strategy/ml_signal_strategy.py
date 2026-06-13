from dataclasses import dataclass
from enum import Enum

from vnpy.common.logger import get_logger
from vnpy.factor.factor_sample import FactorSample


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(slots=True)
class StrategySignal:
    symbol: str
    datetime: str
    action: SignalAction
    reason: str
    score: float = 0.0


class MlSignalStrategy:
    """
    Rule-based signal scaffold for the realtime factor pipeline.

    The class name keeps "ML" because it is intended to be replaced or
    extended by a trained model later. For now it makes deterministic signals
    from momentum, volume and volatility.
    """

    def __init__(
        self,
        min_buy_momentum: float = 0.001,
        min_sell_momentum: float = -0.001,
        max_buy_volatility: float = 0.03,
        high_volume_ratio: float = 1.2,
        enable_log: bool = True,
        enable_print: bool = True,
    ) -> None:
        self.min_buy_momentum = min_buy_momentum
        self.min_sell_momentum = min_sell_momentum
        self.max_buy_volatility = max_buy_volatility
        self.high_volume_ratio = high_volume_ratio
        self.enable_log = enable_log
        self.enable_print = enable_print
        self.logger = get_logger("strategy.ml_signal")

    def on_sample(self, context, sample: FactorSample) -> StrategySignal:
        signal = self.generate_signal(sample)
        # if self.enable_log:
            # self.logger.info(
            #     "signal symbol=%s datetime=%s action=%s score=%.6f reason=%s",
            #     signal.symbol,
            #     signal.datetime,
            #     signal.action.value,
            #     signal.score,
            #     signal.reason,
            # )
        # if self.enable_print:
        #     print(
        #         f"[signal] {signal.datetime} {signal.symbol} "
        #         f"{signal.action.value} score={signal.score:.6f} {signal.reason}"
        #     )
        return signal

    def generate_signal(self, sample: FactorSample) -> StrategySignal:
        score = self._score(sample)

        if self._is_buy(sample):
            return StrategySignal(
                symbol=sample.symbol,
                datetime=sample.datetime,
                action=SignalAction.BUY,
                score=score,
                reason=(
                    "up momentum with acceptable volatility and active volume"
                ),
            )

        if self._is_sell(sample):
            return StrategySignal(
                symbol=sample.symbol,
                datetime=sample.datetime,
                action=SignalAction.SELL,
                score=score,
                reason="down momentum or price down with heavy volume",
            )

        return StrategySignal(
            symbol=sample.symbol,
            datetime=sample.datetime,
            action=SignalAction.HOLD,
            score=score,
            reason="no clear edge",
        )

    def _is_buy(self, sample: FactorSample) -> bool:
        momentum_ok = (
            sample.trend == "UP"
            and sample.momentum >= self.min_buy_momentum
        )
        volume_ok = (
            sample.volume_ratio >= self.high_volume_ratio
            or sample.volume_level == "HIGH_VOLUME"
        )
        volatility_ok = sample.volatility <= self.max_buy_volatility
        return momentum_ok and volume_ok and volatility_ok

    def _is_sell(self, sample: FactorSample) -> bool:
        momentum_bad = (
            sample.trend == "DOWN"
            and sample.momentum <= self.min_sell_momentum
        )
        volume_risk = sample.price_volume_signal == "PRICE_DOWN_VOLUME_UP"
        return momentum_bad or volume_risk

    def _score(self, sample: FactorSample) -> float:
        volume_boost = min(sample.volume_ratio, 3.0) / 3.0
        volatility_penalty = min(sample.volatility, 0.05)
        return sample.momentum + volume_boost * 0.001 - volatility_penalty

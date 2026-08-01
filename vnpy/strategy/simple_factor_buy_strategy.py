from __future__ import annotations

from typing import Any

from vnpy.strategy.strategy_context import SignalDirection, StrategyContext, StrategySignal
from vnpy.strategy.strategy_template import StrategyOutput, StrategyTemplate


class SimpleFactorBuyStrategy(StrategyTemplate):
    """
    Generate a simple buy signal from one factor result.

    This strategy is intentionally small:
        factor_result -> buy condition -> StrategySignal(LONG)

    It does not send orders directly.
    """

    author = "factor"

    def on_init(self, context: StrategyContext) -> None:
        self.factor_name = str(self.setting.get("factor_name", "momentum_20"))
        self.min_return = float(self.setting.get("min_return", 0.002))
        self.min_confidence = float(self.setting.get("min_confidence", 0.5))
        self.emit_repeated = bool(self.setting.get("emit_repeated", False))
        self.volatility_factor_name = str(self.setting.get("volatility_factor_name", "volatility_20"))
        self.max_volatility = float(self.setting.get("max_volatility", 0.03))
        self.volume_factor_name = str(self.setting.get("volume_factor_name", "volume_20"))
        self.min_volume_ratio = float(self.setting.get("min_volume_ratio", 0.0))
        self.volume_price_factor_name = str(self.setting.get("volume_price_factor_name", "price_volume_reversal_20"))
        self.block_bearish_divergence = bool(self.setting.get("block_bearish_divergence", True))
        self.intraday_fade_factor_name = str(self.setting.get("intraday_fade_factor_name", "intraday_fade_20"))
        self.block_intraday_fade = bool(self.setting.get("block_intraday_fade", True))
        self._last_buy_symbols: set[str] = set()

    def on_factor(self, context: StrategyContext, sample: Any, factor_result: Any = None,) -> list[StrategyOutput]:
        factors = self._get_factor_values(factor_result)
        factor_value = factors.get(self.factor_name)
        if factor_value is None or not getattr(factor_value, "is_ready", True):
            return []

        symbol = factor_value.symbol or getattr(sample, "symbol", None)
        if not symbol:
            return []

        reasons: list[str] = []
        fields = factor_value.fields

        ret_n = float(fields.get("ret_n", factor_value.value or 0.0))
        trend = str(fields.get("trend", "UNKNOWN"))
        should_buy = trend == "UP" and ret_n >= self.min_return

        if not should_buy:
            self._last_buy_symbols.discard(symbol)
            return []

        reasons.append(f"{self.factor_name}: trend={trend}, ret_n={ret_n:.6f}")

        if self._blocked_by_volatility(factors, reasons):
            self._last_buy_symbols.discard(symbol)
            return []

        if self._blocked_by_volume(factors, reasons):
            self._last_buy_symbols.discard(symbol)
            return []

        if self._blocked_by_volume_price(factors, reasons):
            self._last_buy_symbols.discard(symbol)
            return []

        if self._blocked_by_intraday_fade(factors, reasons):
            self._last_buy_symbols.discard(symbol)
            return []

        if not self.emit_repeated and symbol in self._last_buy_symbols:
            return []

        confidence = min(1.0, abs(ret_n) / max(abs(self.min_return), 1e-12))
        if confidence < self.min_confidence:
            return []

        self._last_buy_symbols.add(symbol)

        return [
            StrategySignal(
                strategy_name=self.strategy_name,
                symbol=symbol,
                direction=SignalDirection.LONG,
                score=ret_n,
                confidence=confidence,
                reason="; ".join(reasons),
                extra={
                    "source": self.factor_name,
                    "ret_n": ret_n,
                    "trend": trend,
                    "close": float(getattr(sample, "close", 0.0)) if sample is not None else 0.0,
                },
            )
        ]

    @staticmethod
    def _get_factor_values(factor_result: Any) -> dict[str, Any]:
        values = getattr(factor_result, "values", None)
        if not values:
            return {}

        result: dict[str, Any] = {}

        for value in values:
            factor_name = getattr(value, "factor_name", None)
            if factor_name:
                result[factor_name] = value

        return result

    def _blocked_by_volatility(self, factors: dict[str, Any], reasons: list[str]) -> bool:
        factor_value = factors.get(self.volatility_factor_name)
        if factor_value is None:
            return False

        volatility = float(factor_value.value or 0.0)
        reasons.append(f"{self.volatility_factor_name}: volatility={volatility:.6f}")
        return volatility > self.max_volatility

    def _blocked_by_volume(self, factors: dict[str, Any], reasons: list[str]) -> bool:
        factor_value = factors.get(self.volume_factor_name)
        if factor_value is None:
            return False

        volume_ratio = float(factor_value.value or 0.0)
        price_volume_signal = str(factor_value.fields.get("price_volume_signal", "UNKNOWN"))
        reasons.append(
            f"{self.volume_factor_name}: volume_ratio={volume_ratio:.6f}, "
            f"signal={price_volume_signal}"
        )
        return volume_ratio < self.min_volume_ratio

    def _blocked_by_volume_price(self, factors: dict[str, Any], reasons: list[str]) -> bool:
        factor_value = factors.get(self.volume_price_factor_name)
        if factor_value is None:
            return False

        signal = str(factor_value.fields.get("signal", "UNKNOWN"))
        score = float(factor_value.value or 0.0)
        reasons.append(f"{self.volume_price_factor_name}: signal={signal}, score={score:.6f}")
        return self.block_bearish_divergence and signal == "BEARISH_DIVERGENCE"

    def _blocked_by_intraday_fade(self, factors: dict[str, Any], reasons: list[str]) -> bool:
        factor_value = factors.get(self.intraday_fade_factor_name)
        if factor_value is None:
            return False

        signal = bool(factor_value.fields.get("signal", False))
        fade_factor = float(factor_value.value or 0.0)
        reasons.append(f"{self.intraday_fade_factor_name}: signal={signal}, factor={fade_factor:.6f}")
        return self.block_intraday_fade and signal

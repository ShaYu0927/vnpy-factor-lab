from __future__ import annotations

from typing import Any

from vnpy.strategy.StratContext import StrategyContext
from vnpy.strategy.StratTemplate import StrategyOutput, StrategyTemplate


class FactorDebugStrategy(StrategyTemplate):
    """
    Debug strategy for checking whether factor events reach StrategyEngine.

    It only observes incoming factor samples/results and returns no trading
    output, so it will not trigger downstream trade events.
    """

    author = "debug"

    def on_init(self, context: StrategyContext) -> None:
        self.event_count = 0
        self.print_limit = int(self.setting.get("print_limit", 20))
        self.print_factor_values = bool(self.setting.get("print_factor_values", True))
        self.max_factor_values = int(self.setting.get("max_factor_values", 10))
        self.write_log(context, "initialized")

    def on_start(self, context: StrategyContext) -> None:
        self.write_log(context, "started")
        print(f"[{self.strategy_name}] started")

    def on_stop(self, context: StrategyContext) -> None:
        self.write_log(context, "stopped")
        print(f"[{self.strategy_name}] stopped, received={self.event_count}")

    def on_factor(
        self,
        context: StrategyContext,
        sample: Any,
        factor_result: Any = None,
    ) -> list[StrategyOutput]:
        self.event_count += 1

        if self.event_count > self.print_limit:
            return []

        symbol = getattr(sample, "symbol", None)
        dt = getattr(sample, "datetime", None)
        close = getattr(sample, "close", None)
        momentum = getattr(sample, "momentum", None)
        trend = getattr(sample, "trend", None)
        volatility = getattr(sample, "volatility", None)
        volume_ratio = getattr(sample, "volume_ratio", None)

        print(
            f"[{self.strategy_name}] factor event #{self.event_count} "
            f"symbol={symbol} datetime={dt} close={close} "
            f"momentum={momentum} trend={trend} "
            f"volatility={volatility} volume_ratio={volume_ratio}"
        )

        if self.print_factor_values:
            self._print_factor_values(factor_result)

        return []

    def _print_factor_values(self, factor_result: Any) -> None:
        values = getattr(factor_result, "values", None)
        if not values:
            print(f"[{self.strategy_name}] factor_result.values is empty")
            return

        for value in list(values)[: self.max_factor_values]:
            factor_name = getattr(value, "factor_name", None)
            factor_value = getattr(value, "value", None)
            symbol = getattr(value, "symbol", None)
            print(
                f"[{self.strategy_name}]   {symbol} "
                f"{factor_name}={factor_value}"
            )

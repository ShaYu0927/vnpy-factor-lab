from __future__ import annotations

from abc import ABC
from typing import Any, Sequence, Union
from vnpy.strategy.StratContext import StrategyContext, TargetPosition, StrategySignal


StrategyOutput = Union[StrategySignal, TargetPosition]
class StrategyTemplate(ABC):
    """
    Base class for strategies managed by StrategyEngine.

    Responsibilities:
        1. Manage strategy lifecycle.
        2. Receive events.
        3. Make trading decisions.
        4. Return signals or target positions.

    Strategy should not send orders directly.
    Order generation, risk check and execution should be handled by
    PortfolioEngine, RiskEngine and ExecutionEngine.
    """

    author = ""

    def __init__(
        self,
        strategy_engine: Any,
        strategy_name: str,
        symbols: Sequence[str],
        factor_names: Sequence[str] | None = None,
        setting: dict[str, Any] | None = None,
    ) -> None:
        self.strategy_engine = strategy_engine
        self.strategy_name = strategy_name
        self.symbols = list(symbols)
        self.factor_names = list(factor_names or [])
        self.setting = setting or {}

        self.inited = False
        self.trading = False

    def on_init(self, context: StrategyContext) -> None:
        """
        Called once before the strategy starts.
        """

    def on_start(self, context: StrategyContext) -> None:
        """
        Called when the strategy is started.
        """

    def on_stop(self, context: StrategyContext) -> None:
        """
        Called when the strategy is stopped.
        """

    def on_bar(
        self,
        context: StrategyContext,
        bar: Any,
    ) -> list[StrategyOutput]:
        """
        Handle bar event.

        Return strategy signals or target positions.
        """
        return []

    def on_factor(
        self,
        context: StrategyContext,
        sample: Any,
        factor_result: Any = None,
    ) -> list[StrategyOutput]:
        """
        Handle factor event.

        Return strategy signals or target positions.
        """
        return []

    def on_signal(
        self,
        context: StrategyContext,
        signal: StrategySignal,
    ) -> list[StrategyOutput]:
        """
        Handle upstream strategy signal event.
        """
        return []

    def on_timer(
        self,
        context: StrategyContext,
    ) -> list[StrategyOutput]:
        """
        Handle timer event.

        This is useful for daily rebalancing, tail-close strategy,
        scheduled factor calculation, etc.
        """
        return []

    def on_order(
        self,
        context: StrategyContext,
        order: Any,
    ) -> list[StrategyOutput]:
        """
        Handle order update event.

        Usually used to update internal state.
        """
        return []

    def on_trade(
        self,
        context: StrategyContext,
        trade: Any,
    ) -> list[StrategyOutput]:
        """
        Handle trade update event.

        Usually used to update internal state.
        """
        return []

    def write_log(self, context: StrategyContext, msg: str) -> None:
        """
        Write strategy log.

        Logger can be injected through context.params or handled by StrategyEngine.
        """
        logger = context.params.get("logger")

        if logger:
            logger.info("[%s] %s", self.strategy_name, msg)

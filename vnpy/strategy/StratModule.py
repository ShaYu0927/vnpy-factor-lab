from __future__ import annotations

from typing import Any

from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType
from vnpy.strategy.StratEngine import StrategyEngine


class StrategyEngineModule(BaseModule):
    """
    ModuleEngine adapter for StrategyEngine.

    Register it with ModuleEngine:
        module_engine.register_module("strategy", strategy_engine_module_entry)
    """

    def handle(self, event: EngineEvent) -> None:
        engine = self.strategy_engine

        if event.event_type == EventType.STOP:
            engine.stop_all()
            return

        engine.on_event(event)
        if engine.latest_outputs:
            output = engine.latest_outputs[-1]
            self.set_state("latest_output", output)
            self.set_state("latest_symbol", output.symbol)
            self.set_state("latest_strategy", output.strategy_name)

            if event.event_type == EventType.FACTOR:
                self.set_state("latest_signal", output)

    @property
    def strategy_engine(self) -> StrategyEngine:
        engine = self.get_object("strategy_engine")
        if engine is not None:
            return engine

        engine = StrategyEngine(
            post_event=self.ctx.engine.post_event,
            source=self.name,
            order_module=self.get_config("order_module", "order"),
            trade_module=self.get_config("trade_module"),
            risk_engine=self.get_object("risk_engine"),
            risk_context=self.get_object("risk_context"),
        )
        self.set_object("strategy_engine", engine)
        self._load_configured_strategies(engine)
        return engine

    def _load_configured_strategies(self, engine: StrategyEngine) -> None:
        strategies = self.get_config("strategies", [])

        for config in strategies:
            self._add_configured_strategy(engine, config)

    def _add_configured_strategy(
        self,
        engine: StrategyEngine,
        config: dict[str, Any],
    ) -> None:
        strategy_name = config["name"]
        strategy_class = config["class"]
        symbols = config.get("symbols", [])
        factor_names = config.get("factors", [])
        setting = config.get("setting", {})
        active = bool(config.get("active", True))

        engine.add_strategy(
            strategy_name=strategy_name,
            strategy_class=strategy_class,
            symbols=symbols,
            factor_names=factor_names,
            setting=setting,
            auto_init=True,
        )

        if active:
            engine.start_strategy(strategy_name)


strategy_engine_module_entry = make_module_entry(StrategyEngineModule)

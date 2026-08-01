from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType
from vnpy.strategy.ml_signal_strategy import MlSignalStrategy


class SignalStrategyModule(BaseModule):
    """
    ModuleEngine module for factor-sample driven signal generation.
    """

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.FACTOR:
            return

        sample = event.get("sample")
        if sample is None:
            return

        signal = self.strategy.on_sample(None, sample)

        self.set_state("latest_signal", signal)
        self.set_state("latest_symbol", signal.symbol)
        self.set_state("latest_datetime", signal.datetime)
        self.set_state("latest_action", signal.action.value)

        target = self.get_config("trade_module")
        if not target:
            return

        self.post(
            target=target,
            event_type=EventType.TRADE_SIGNAL,
            symbol=signal.symbol,
            data={"signal": signal},
        )

    @property
    def strategy(self) -> MlSignalStrategy:
        strategy = self.get_object("strategy")
        if strategy is not None:
            return strategy

        strategy = MlSignalStrategy(
            min_buy_momentum=float(self.get_config("min_buy_momentum", 0.001)),
            min_sell_momentum=float(self.get_config("min_sell_momentum", -0.001)),
            max_buy_volatility=float(self.get_config("max_buy_volatility", 0.03)),
            high_volume_ratio=float(self.get_config("high_volume_ratio", 1.2)),
            enable_log=bool(self.get_config("enable_log", True)),
            enable_print=bool(self.get_config("enable_print", True)),
        )

        self.set_object("strategy", strategy)
        return strategy


strategy_module_entry = make_module_entry(SignalStrategyModule)

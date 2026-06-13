from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.factorEngine import FactorBatchResult, FactorValue
from vnpy.factor.factor_sample import FactorSample
from vnpy.strategy.StratContext import SignalDirection
from vnpy.strategy.StratEngine import StrategyEngine
from vnpy.strategy.StratModule import strategy_engine_module_entry
from vnpy.strategy.StratTemplate import StrategyTemplate


def create_buy_sample() -> FactorSample:
    return FactorSample(
        symbol="SZSE.000001",
        datetime="2026-05-25 10:01:00",
        close=12.5,
        momentum=0.01,
        strength=0.8,
        trend="UP",
        volatility=0.01,
        latest_volume=2000,
        volume_ma5=1000,
        volume_ma10=900,
        volume_ratio=2.0,
        volume_change=1.0,
        volume_level="HIGH_VOLUME",
        price_volume_signal="PRICE_UP_VOLUME_UP",
        momentum_reason="up",
        volume_reason="active",
    )


def create_factor_result(*factor_names: str) -> FactorBatchResult:
    return FactorBatchResult(
        values=[
            FactorValue(
                symbol="SZSE.000001",
                factor_name=factor_name,
                value=1.0,
            )
            for factor_name in factor_names
        ]
    )


def test_factor_signal_strategy_runs_through_strategy_engine_module() -> None:
    received = []

    def trade_entry(ctx, event) -> None:
        received.append(event)

    engine = ModuleEngine()
    engine.register_module(name="trade", entry=trade_entry)
    engine.register_module(
        name="strategy",
        entry=strategy_engine_module_entry,
        config={
            "trade_module": "trade",
            "strategies": [
                {
                    "name": "factor_signal",
                    "class": "vnpy.strategy.factor_signal_strategy.FactorSignalStrategy",
                    "active": True,
                    "factors": ["momentum_20", "volatility_20", "volume_20"],
                    "setting": {"enable_log": False, "enable_print": False},
                }
            ],
        },
    )
    engine.start_all()

    engine.post_event(
        target="strategy",
        event=EngineEvent(
            event_type=EventType.FACTOR,
            source="factor",
            symbol="SZSE.000001",
            data={
                "sample": create_buy_sample(),
                "factor_result": create_factor_result(
                    "momentum_20",
                    "volatility_20",
                    "volume_20",
                ),
            },
        ),
    )

    engine.get_module("strategy")._queue.join()
    engine.get_module("trade")._queue.join()

    context = engine.get_context("strategy")
    strategy_engine = context.get_object("strategy_engine")
    signal = context.get_state("latest_signal")

    assert "factor_signal" in strategy_engine.active_strategies
    assert signal.direction == SignalDirection.LONG
    assert len(received) == 1
    assert received[0].event_type == EventType.TRADE_SIGNAL
    assert received[0].get("signal").direction == SignalDirection.LONG

    engine.stop_all()


class RecordingStrategy(StrategyTemplate):
    def on_init(self, context) -> None:
        self.received = []

    def on_factor(self, context, sample, factor_result=None):
        self.received.append(factor_result)
        return []


def test_factor_event_only_dispatches_to_mapped_strategies() -> None:
    engine = StrategyEngine(post_event=lambda target, event: True)
    momentum_strategy = engine.add_strategy(
        strategy_name="momentum",
        strategy_class=RecordingStrategy,
        factor_names=["momentum_20"],
    )
    volume_strategy = engine.add_strategy(
        strategy_name="volume",
        strategy_class=RecordingStrategy,
        factor_names=["volume_20"],
    )
    complete_strategy = engine.add_strategy(
        strategy_name="combined",
        strategy_class=RecordingStrategy,
        factor_names=["momentum_20", "volume_20"],
    )
    engine.start_all()

    engine.on_factor(
        sample=create_buy_sample(),
        factor_result=create_factor_result("momentum_20"),
        symbol="SZSE.000001",
    )

    assert len(momentum_strategy.received) == 1
    assert len(volume_strategy.received) == 0
    assert len(complete_strategy.received) == 0

    engine.on_factor(
        sample=create_buy_sample(),
        factor_result=create_factor_result("momentum_20", "volume_20"),
        symbol="SZSE.000001",
    )

    assert len(momentum_strategy.received) == 2
    assert len(volume_strategy.received) == 1
    assert len(complete_strategy.received) == 1

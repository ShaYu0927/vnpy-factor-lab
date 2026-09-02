from vnpy.event.context import ModuleContext
from vnpy.factor.realtime_module import RealtimeFactorModule


def test_factor_module_retains_legacy_strategy_target() -> None:
    context = ModuleContext("factor")
    context.set_config("strategy_module", "legacy_strategy")

    module = RealtimeFactorModule(context)

    assert module.event_targets == ("legacy_strategy",)


def test_factor_module_supports_polymorphic_event_consumers() -> None:
    context = ModuleContext("factor")
    context.set_config(
        "factor_targets",
        ["model", "strategy", "recorder", "strategy"],
    )

    module = RealtimeFactorModule(context)

    assert module.event_targets == ("model", "strategy", "recorder")

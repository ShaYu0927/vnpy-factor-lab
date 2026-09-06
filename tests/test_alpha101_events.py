from datetime import datetime, timedelta

import pytest

from vnpy.datafeed.model import MarketBar
from vnpy.event.context import ModuleContext
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_module import RealtimeFactorModule


class EventSink:
    def __init__(self):
        self.events = []

    def post_event(self, target, event):
        self.events.append((target, event))
        return True


def make_module(factors=(33, 56, 101)):
    sink = EventSink()
    context = ModuleContext("factor", sink)
    context.config.update(frequency="1d", universe=["A", "B"],
                          alpha101_factors=list(factors), factor_targets=["strategy", "recorder"],
                          enable_print=False)
    return RealtimeFactorModule(context), sink


def event(symbol, day=0, frequency="1d"):
    opening, closing = (10, 11) if symbol == "A" else (20, 19)
    return EngineEvent(EventType.BAR, {"bar": MarketBar(
        symbol, datetime(2025, 1, 1) + timedelta(days=day),
        opening, max(opening, closing) + 1, min(opening, closing) - 1,
        closing, 1000, frequency=frequency,
    )})


def test_event_barrier_fanout_and_invalid_fields():
    module, sink = make_module()
    module.handle(event("A"))
    assert sink.events == []
    module.handle(event("B"))
    assert len(sink.events) == 4
    assert {(target, e.symbol) for target, e in sink.events} == {
        (target, symbol) for target in ("strategy", "recorder") for symbol in ("A", "B")}
    sample = sink.events[0][1].get("sample")
    assert sample.features["alpha033"] == pytest.approx(0.5)
    assert sample.features["alpha101"] == pytest.approx(1 / 3.001)
    assert "alpha056" not in sample.features
    invalid = sink.events[0][1].get("factor_result").get("A", "alpha056")
    assert invalid.value is None and not invalid.is_ready and invalid.reason
    module.handle(event("B"))
    module.handle(event("A", frequency="60s"))
    module.handle(event("OUTSIDE"))
    assert len(sink.events) == 4
    module.handle(event("B", day=1))
    assert len(sink.events) == 4
    module.handle(event("A", day=1))
    assert len(sink.events) == 8
    module.handle(event("A", day=0))
    assert len(sink.events) == 8


def test_all_101_are_present_in_event_result():
    module, sink = make_module(range(1, 102))
    module.handle(event("A"))
    module.handle(event("B"))
    assert len(sink.events) == 4
    assert len(sink.events[0][1].get("factor_result").values) == 202


def test_alpha101_requires_universe_and_daily_frequency():
    module, _ = make_module()
    module.ctx.config["universe"] = []
    with pytest.raises(ValueError, match="universe"):
        _ = module.factor_service
    module.ctx.config["universe"] = ["A", "B"]
    module.ctx.config["frequency"] = "60s"
    with pytest.raises(ValueError, match="daily"):
        _ = module.factor_service

"""Offline event-system demo: python -m examples.alpha101_events.

Feed only completed daily BAR events. The configured universe must finish the
same timestamp before calculation. Duplicate/late bars are ignored after
publication; this service does not retract previously published results.
Historical bars can be loaded into module.factor_service.bar_cache before replay.
Missing/undefined factors remain in factor_result with value=None; sample.features
contains finite factors only. Check required features before making decisions.
"""
import logging
from datetime import datetime

from vnpy.common.logger import init_global_logger, get_logger, shutdown_global_logger
from vnpy.datafeed.model import MarketBar
from vnpy.event.context import ModuleContext
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_module import RealtimeFactorModule


ALPHA101_CONFIG = {
    "frequency": "1d",
    "universe": ["A", "B"],
    "alpha101_factors": list(range(1, 102)),
    "alpha101_history": 320,
    "maxlen": 30000,
    "factor_targets": ["recorder"],
    "enable_print": False,
}
# Use this config with ModuleEngine.register_module(entry=factor_module_entry, ...).
# Replace A/B with the actual subscription universe. Do not mix expression
# `alphas` into the same module. Pass vwap/market_cap/industry/sector/subindustry
# as date-appropriate bar.extra fields when available.


class LoggingConsumer:
    def post_event(self, target, event):
        sample = event.get("sample")
        get_logger("alpha.events").info(
            "FACTOR delivered: target=%s symbol=%s finite_features=%s",
            target, sample.symbol, dict(sample.features),
        )
        return True


def main():
    init_global_logger(level=logging.DEBUG)
    try:
        context = ModuleContext("factor", LoggingConsumer())
        context.config.update(ALPHA101_CONFIG)
        module = RealtimeFactorModule(context)
        for symbol, opening, closing in (("A", 10, 11), ("B", 20, 19)):
            bar = MarketBar(symbol, datetime(2025, 1, 1), opening,
                            max(opening, closing) + 1, min(opening, closing) - 1,
                            closing, 1000, frequency="1d")
            module.handle(EngineEvent(EventType.BAR, {"bar": bar}))
    finally:
        shutdown_global_logger()


if __name__ == "__main__":
    main()

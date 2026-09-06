from vnpy.datafeed.bar_cache import BarCache
from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.engine import AlphaSampleCache
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.realtime_service import RealtimeAlphaService
from vnpy.alpha.logger import logger


class RealtimeFactorModule(BaseModule):
    """
    ModuleEngine module for realtime factor calculation.
    """

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.BAR:
            return

        service = self.factor_service
        bar = event.get("bar")
        service.on_bar(bar)
        for sample in service.latest_samples:
            self._publish_sample(event, service, sample)

    def _publish_sample(self, event, service, sample) -> None:
        self.set_state("latest_sample", sample)
        self.set_state("latest_symbol", sample.symbol)
        self.set_state("latest_datetime", sample.datetime)
        self.set_state("latest_factor_result", service.latest_batch_result)

        if self.get_config("enable_print", True):
            event_count = int(self.get_state("factor_event_count", 0)) + 1
            self.set_state("factor_event_count", event_count)
            print_every = max(1, int(self.get_config("print_every", 20)))
            if event_count % print_every == 0:
                self._print_factor_event(
                    sample,
                    service.latest_batch_result,
                    event_count,
                )

        data = {
            "sample": sample,
            "factor_result": service.latest_batch_result,
            "bar_event_id": event.event_id,
        }
        for target in self.event_targets:
            posted = self.post(
                target=target,
                event_type=EventType.FACTOR,
                symbol=sample.symbol,
                data=dict(data),
            )
            logger.info("FACTOR event: target=%s symbol=%s at=%s features=%d posted=%s",
                        target, sample.symbol, sample.datetime, len(sample.features), posted)

    @property
    def event_targets(self) -> tuple[str, ...]:
        """Resolve factor consumers while retaining strategy_module compatibility."""
        configured = self.get_config("factor_targets")
        if configured is None:
            configured = [self.get_config("strategy_module", "strategy")]
        elif isinstance(configured, str):
            configured = [configured]
        if not isinstance(configured, (list, tuple)):
            raise TypeError("factor_targets must be a string or sequence of strings")
        targets = tuple(dict.fromkeys(str(item).strip() for item in configured if str(item).strip()))
        if not targets:
            raise ValueError("factor_targets must not be empty")
        return targets

    @property
    def factor_service(self) -> RealtimeAlphaService:
        service = self.get_object("factor_service")
        if service is not None:
            return service

        maxlen = int(self.get_config("maxlen", 30000))
        frequency = self.get_config("frequency", "60s")
        bar_cache = BarCache(maxlen=maxlen)
        sample_cache = AlphaSampleCache(maxlen=maxlen)
        raw_definitions = self.get_config("alphas", [])
        definitions = tuple(AlphaDefinition(**item) for item in raw_definitions)
        universe = self.get_config("universe")
        service = RealtimeAlphaService(
            bar_cache=bar_cache,
            sample_cache=sample_cache,
            definitions=definitions,
            universe=universe,
            frequency=frequency,
            alpha101_factors=self.get_config("alpha101_factors"),
            alpha101_history=int(self.get_config("alpha101_history", 320)),
        )

        self.set_object("bar_cache", bar_cache)
        self.set_object("sample_cache", sample_cache)
        self.set_object("factor_service", service)
        return service

    def _print_factor_event(self, sample, factor_result, event_count: int) -> None:
        values = getattr(factor_result, "values", []) or []
        errors = getattr(factor_result, "errors", []) or []
        factor_values = []

        for value in values:
            factor_name = getattr(value, "factor_name", "")
            display_value = getattr(value, "value", None)
            if factor_name and display_value is not None:
                factor_values.append(f"{factor_name}={display_value:.6f}")

        print(
            f"[factor] #{event_count} "
            f"symbol={sample.symbol} "
            f"datetime={sample.datetime} "
            f"{' '.join(factor_values)} "
            f"errors={len(errors)}"
        )

factor_module_entry = make_module_entry(RealtimeFactorModule)

from vnpy.datafeed.data_bar_cache import BarCache
from vnpy.alpha.definition import AlphaDefinition
from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine, AlphaSampleCache
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.factor_realtime_service import RealtimeAlphaService
from vnpy.factor.core.factor_engine import FactorBatchResult
from vnpy.alpha.logger import logger


class RealtimeFactorModule(BaseModule):
    """
    ModuleEngine module for realtime factor calculation.
    """

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.BAR:
            return

        service = self.factor_service
        bars = event.get("bars")
        if bars is None:
            service.on_bar(event.get("bar"))
            bars = [event.get("bar")]
        else:
            service.on_bars(bars)
        self.set_state("processed_bars", self.get_state("processed_bars", 0) + len(bars))
        self.set_state("calculation_batches", self.get_state("calculation_batches", 0) + 1)
        values_by_symbol = {}
        for value in service.latest_batch_result.values:
            values_by_symbol.setdefault(value.symbol, []).append(value)
        for sample in service.latest_samples:
            result = FactorBatchResult(values=values_by_symbol.get(sample.symbol, []))
            self._publish_sample(event, sample, result)

    def _publish_sample(self, event, sample, result) -> None:
        self.set_state("sample_count", self.get_state("sample_count", 0) + 1)
        self.set_state("latest_sample", sample)
        self.set_state("latest_symbol", sample.symbol)
        self.set_state("latest_datetime", sample.datetime)
        self.set_state("latest_factor_result", result)

        if self.get_config("enable_print", True):
            event_count = int(self.get_state("factor_event_count", 0)) + 1
            self.set_state("factor_event_count", event_count)
            print_every = max(1, int(self.get_config("print_every", 20)))
            if event_count % print_every == 0:
                self._print_factor_event(
                    sample,
                    result,
                    event_count,
                )

        data = {
            "sample": sample,
            "factor_result": result,
            "bar_event_id": event.event_id,
        }
        for target in self.event_targets:
            posted = self.post(
                target=target,
                event_type=EventType.FACTOR,
                symbol=sample.symbol,
                data=dict(data),
            )
            if not posted:
                raise RuntimeError(f"factor sample rejected by {target}: {sample.symbol} {sample.datetime}")
            logger.debug("FACTOR event: target=%s symbol=%s at=%s features=%d posted=%s",
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

        frequency = self.get_config("frequency", "60s")
        raw_definitions = self.get_config("alphas", [])
        definitions = tuple(
            Alpha(**item) if "formula" in item else AlphaDefinition(**item)
            for item in raw_definitions
        )
        alpha_engine = AlphaEngine(definitions)
        # History for formula evaluation and history for strategies have different needs.
        bar_cache = BarCache(maxlen=max(alpha_engine.min_bars, int(self.get_config("bar_maxlen", 0))))
        sample_cache = AlphaSampleCache(maxlen=int(self.get_config("sample_maxlen", self.get_config("maxlen", 1000))))
        universe = self.get_config("universe")
        service = RealtimeAlphaService(
            bar_cache=bar_cache,
            sample_cache=sample_cache,
            definitions=definitions,
            alpha_engine=alpha_engine,
            universe=universe,
            frequency=frequency,
            optional_names=self.get_config("optional_alphas", ()),
        )

        self.set_object("bar_cache", bar_cache)
        self.set_object("sample_cache", sample_cache)
        self.set_object("factor_service", service)
        return service

    def _print_factor_event(self, sample, factor_result, event_count: int) -> None:
        errors = getattr(factor_result, "errors", []) or []
        factor_values = []

        # 批次可能包含多只股票，只打印当前样本对应的因子值。
        for factor_name, value in sample.features.items():
            display_value = f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
            factor_values.append(f"{factor_name}={display_value}")

        print(
            f"[factor] #{event_count} "
            f"symbol={sample.symbol} "
            f"datetime={sample.datetime} "
            f"{' '.join(factor_values)} "
            f"errors={len(errors)}",
            flush=True,
        )

factor_module_entry = make_module_entry(RealtimeFactorModule)

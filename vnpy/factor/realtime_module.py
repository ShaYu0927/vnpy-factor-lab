from vnpy.datafeed.BarCache import BarCache
from vnpy.event.base_module import BaseModule, make_module_entry
from vnpy.event.event import EngineEvent, EventType
from vnpy.factor.core.factorEngine import ExecutionMode
from vnpy.factor.core.factor_sample import FastFactorSampleCache
from vnpy.factor.realtime_service import RealtimeFactorService


class RealtimeFactorModule(BaseModule):
    """
    ModuleEngine module for realtime factor calculation.
    """

    def handle(self, event: EngineEvent) -> None:
        if event.event_type != EventType.BAR:
            return

        service = self.factor_service
        bar = event.get("bar")
        sample = service.on_bar(bar)

        if sample is None:
            return

        self.set_state("latest_sample", sample)
        self.set_state("latest_symbol", sample.symbol)
        self.set_state("latest_datetime", sample.datetime)
        self.set_state("latest_factor_result", service.latest_batch_result)

        self.post(
            target=self.get_config("strategy_module", "strategy"),
            event_type=EventType.FACTOR,
            symbol=sample.symbol,
            data={
                "sample": sample,
                "factor_result": service.latest_batch_result,
            },
        )

    @property
    def factor_service(self) -> RealtimeFactorService:
        service = self.get_object("factor_service")
        if service is not None:
            return service

        maxlen = int(self.get_config("maxlen", 30000))
        frequency = self.get_config("frequency", "60s")
        mode = ExecutionMode(self.get_config("mode", ExecutionMode.SYNC.value))
        max_workers = self.get_config("max_workers")
        if max_workers is not None:
            max_workers = int(max_workers)

        bar_cache = BarCache(maxlen=maxlen)
        sample_cache = FastFactorSampleCache(maxlen=maxlen)
        service = RealtimeFactorService(
            bar_cache=bar_cache,
            sample_cache=sample_cache,
            frequency=frequency,
            mode=mode,
            max_workers=max_workers,
        )

        self.set_object("bar_cache", bar_cache)
        self.set_object("sample_cache", sample_cache)
        self.set_object("factor_service", service)
        return service


factor_module_entry = make_module_entry(RealtimeFactorModule)

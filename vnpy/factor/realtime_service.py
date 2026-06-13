from typing import Mapping, Optional, Sequence

from vnpy.datafeed.BarCache import BarCache
from vnpy.factor.core.basic_factor import BasicFactorResult
from vnpy.factor.core.factorDataBuilder import (
    BasicMomentumEngineFactor,
    BasicVolatilityEngineFactor,
    BasicVolumeEngineFactor,
)
from vnpy.factor.core.factorEngine import (
    ExecutionMode,
    Factor,
    FactorBatchResult,
    FactorContext,
    FactorEngine,
)
from vnpy.factor.core.factor_sample import FactorSample, FactorSampleBuilder, FastFactorSampleCache


class BasicFactorSet:
    """
    Factory for the default bar-based factors used by realtime calculation.
    """

    @staticmethod
    def create() -> tuple[Factor, ...]:
        return (
            BasicMomentumEngineFactor(window=20),
            BasicVolatilityEngineFactor(window=20),
            BasicVolumeEngineFactor(window=20),
        )


class FactorBatchCalculator:
    """
    Thin batch/cross-section facade around FactorEngine.
    """

    def __init__(self, factor_engine: FactorEngine) -> None:
        self.factor_engine = factor_engine

    def calculate_many(
        self,
        symbol_data_map: Mapping[str, object],
        context: Optional[FactorContext] = None,
    ) -> FactorBatchResult:
        return self.factor_engine.calculate_many(symbol_data_map, context=context)

    def calculate_cross_section(
        self,
        factor_name: str,
        symbol_data_map: Mapping[str, object],
        context: Optional[FactorContext] = None,
    ) -> FactorBatchResult:
        return self.factor_engine.calculate_factor_cross_section(
            factor_name=factor_name,
            symbol_data_map=symbol_data_map,
            context=context,
        )


class CachedBarFactorCalculator:
    """
    Reads latest bars from BarCache and delegates factor calculation to FactorEngine.
    """

    def __init__(self, bar_cache: BarCache, factor_engine: FactorEngine, frequency: str, min_bars: int,) -> None:
        self.bar_cache = bar_cache
        self.factor_engine = factor_engine
        self.frequency = frequency
        self.min_bars = min_bars

    def calculate_latest(self, bar, context: Optional[FactorContext] = None,) -> Optional[FactorBatchResult]:
        history_bars = self.bar_cache.get_bars(symbol=bar.symbol, frequency=bar.frequency, count=self.min_bars,)
        if len(history_bars) < self.min_bars:
            return None

        return self.factor_engine.calculate_one(
            symbol=bar.symbol,
            data=history_bars,
            context=context or FactorContext(trade_date=str(getattr(bar, "bob", ""))),
        )

    def build_latest_data_map(self, symbols: Sequence[str], count: Optional[int] = None,) -> dict[str, object]:
        required_count = count or self.min_bars
        symbol_data_map: dict[str, object] = {}

        for symbol in symbols:
            bars = self.bar_cache.get_bars(
                symbol=symbol,
                frequency=self.frequency,
                count=required_count,
            )
            if len(bars) >= required_count:
                symbol_data_map[symbol] = bars

        return symbol_data_map


class FactorSampleAssembler:
    """
    Converts FactorEngine output into the strategy-facing FactorSample.
    """

    def build_sample(self, bar, batch_result: FactorBatchResult,) -> Optional[FactorSample]:
        result = self.to_basic_result(bar.symbol, batch_result)

        return FactorSampleBuilder.build(
            bar=bar,
            momentum_result=result.momentum,
            volume_result=result.volume,
            volatility_result=result.volatility,
        )

    @staticmethod
    def to_basic_result(symbol: str, batch_result: FactorBatchResult) -> BasicFactorResult:
        result = BasicFactorResult(symbol=symbol)

        for value in batch_result.values:
            if value.factor_name.startswith("momentum_"):
                result.momentum = value.value
            elif value.factor_name.startswith("volatility_"):
                result.volatility = value.value
            elif value.factor_name.startswith("volume_"):
                result.volume = value.value

        return result


class RealtimeFactorService:
    """
    Unified factor service for realtime and batch factor calculation.

    Realtime:
        on_bar() updates BarCache, calculates the latest symbol through FactorEngine
        and returns a FactorSample for the strategy module.

    Batch/cross-section:
        calculate_many() and calculate_cross_section() expose the same FactorEngine
        for offline multi-stock calculation.
    """

    def __init__(
        self,
        bar_cache: BarCache,
        sample_cache: FastFactorSampleCache,
        frequency: str = "60s",
        factors: Optional[Sequence[Factor]] = None,
        factor_engine: Optional[FactorEngine] = None,
        mode: ExecutionMode = ExecutionMode.SYNC,
        max_workers: Optional[int] = None,
        batch_calculator: Optional[FactorBatchCalculator] = None,
        cached_calculator: Optional[CachedBarFactorCalculator] = None,
        sample_assembler: Optional[FactorSampleAssembler] = None,
    ) -> None:
        self.bar_cache = bar_cache
        self.sample_cache = sample_cache
        self.frequency = frequency
        self.factors = tuple(factors or BasicFactorSet.create())
        self.factor_engine = factor_engine or FactorEngine(
            factors=self.factors,
            mode=mode,
            max_workers=max_workers,
        )
        self.min_bars = max((getattr(factor, "min_bars", 1) for factor in self.factors), default=1)
        self.batch_calculator = batch_calculator or FactorBatchCalculator(self.factor_engine)
        self.cached_calculator = cached_calculator or CachedBarFactorCalculator(
            bar_cache=self.bar_cache,
            factor_engine=self.factor_engine,
            frequency=self.frequency,
            min_bars=self.min_bars,
        )
        self.sample_assembler = sample_assembler or FactorSampleAssembler()
        self.latest_batch_result = FactorBatchResult()

    def on_bar(self, bar) -> Optional[FactorSample]:
        """
        Update cache, calculate factors and return the latest realtime sample.
        """

        if bar is None:
            return None

        if not getattr(bar, "frequency", None):
            bar.frequency = self.frequency

        self.bar_cache.update(bar)

        batch_result = self.cached_calculator.calculate_latest(bar)
        if batch_result is None:
            return None

        self.latest_batch_result = batch_result
        sample = self.sample_assembler.build_sample(bar, batch_result)

        if sample is None:
            return None

        self.sample_cache.add(sample)
        return sample

    def calculate_many(
        self,
        symbol_data_map: Mapping[str, object],
        context: Optional[FactorContext] = None,
    ) -> FactorBatchResult:
        """
        Calculate all configured factors for many symbols.

        This is the batch/multi-stock entry point for offline jobs or
        cross-sectional strategy preparation.
        """

        return self.batch_calculator.calculate_many(symbol_data_map, context=context)

    def calculate_cross_section(
        self,
        factor_name: str,
        symbol_data_map: Mapping[str, object],
        context: Optional[FactorContext] = None,
    ) -> FactorBatchResult:
        """
        Calculate one factor across many symbols.
        """

        return self.batch_calculator.calculate_cross_section(
            factor_name=factor_name,
            symbol_data_map=symbol_data_map,
            context=context,
        )

    def calculate_latest_cross_section(
        self,
        symbols: Sequence[str],
        factor_name: Optional[str] = None,
        count: Optional[int] = None,
        context: Optional[FactorContext] = None,
    ) -> FactorBatchResult:
        """
        Calculate factors from the latest cached bars of many symbols.
        """

        symbol_data_map = self.cached_calculator.build_latest_data_map(
            symbols=symbols,
            count=count,
        )

        if factor_name:
            return self.calculate_cross_section(
                factor_name=factor_name,
                symbol_data_map=symbol_data_map,
                context=context,
            )

        return self.calculate_many(symbol_data_map, context=context)

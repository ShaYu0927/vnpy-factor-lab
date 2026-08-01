from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

from vnpy.factor.core.factor_data_builder import (
    BasicMomentumEngineFactor,
    BasicVolatilityEngineFactor,
    BasicVolumeEngineFactor,
)
from vnpy.factor.factor_engine import (
    Factor,
    FactorBatchResult,
    FactorContext,
    FactorEngine,
    FactorOutput,
    FactorStatus,
    FactorValue,
)
from vnpy.factor.realtime_service import FactorSampleAssembler
from vnpy.strategy.strategy_context import SignalDirection, StrategyContext
from vnpy.strategy.simple_factor_buy_strategy import SimpleFactorBuyStrategy


@dataclass
class LegacyResult:
    symbol: str
    score: float
    signal: str
    reason: str


class LegacyFactor(Factor):
    name = "legacy_score"
    primary_field = "score"

    def calculate(self, symbol, data, context):
        return LegacyResult(
            symbol=symbol,
            score=0.75,
            signal="LONG",
            reason="legacy result normalized",
        )


class CanonicalFactor(Factor):
    name = "canonical_score"

    def calculate(self, symbol, data, context):
        return FactorOutput(
            value=-0.25,
            fields={"signal": "SHORT", "confirmed": True},
            reason="canonical result",
        )


class EmptyFactor(Factor):
    name = "empty_score"

    def calculate(self, symbol, data, context):
        return None


def test_legacy_factor_result_is_normalized() -> None:
    result = FactorEngine([LegacyFactor()]).calculate_one(
        "000001.SSE",
        data=[1],
        context=FactorContext(trade_date="2026-07-20 10:00:00"),
    )

    value = result.get("000001.SSE", "legacy_score")

    assert value is not None
    assert value.is_ready
    assert value.value == 0.75
    assert value.primary_field == "score"
    assert value.fields == {"signal": "LONG"}
    assert value.reason == "legacy result normalized"
    assert isinstance(value.raw_value, LegacyResult)


def test_canonical_factor_output_and_batch_helpers() -> None:
    result = FactorEngine([CanonicalFactor()]).calculate_one(
        "000001.SSE",
        data=[1],
    )

    value = result.get("000001.SSE", "canonical_score")

    assert value is not None
    assert value.value == -0.25
    assert value.fields == {"signal": "SHORT", "confirmed": True}
    assert result.for_symbol("000001.SSE") == {"canonical_score": value}
    assert result.scalar_map("000001.SSE") == {"canonical_score": -0.25}
    assert result.to_records()[0]["status"] == "ready"
    assert "raw_value" not in result.to_records()[0]


def test_factor_value_accepts_legacy_extra_fields() -> None:
    value = FactorValue(
        symbol="000001.SSE",
        factor_name="legacy",
        value=1.0,
        extra={"signal": "LONG"},
    )

    assert value.fields == {"signal": "LONG"}
    assert value.extra is value.fields


def test_none_result_has_explicit_insufficient_status() -> None:
    result: FactorBatchResult = FactorEngine([EmptyFactor()]).calculate_one(
        "000001.SSE",
        data=[1],
    )
    value = result.get("000001.SSE", "empty_score")

    assert value is not None
    assert value.value is None
    assert value.status == FactorStatus.INSUFFICIENT
    assert not value.is_ready


def test_builtin_factors_publish_scalars_and_build_strategy_sample() -> None:
    symbol = "000001.SSE"
    bars = [
        SimpleNamespace(close=10 + index, volume=100 + index * 10)
        for index in range(4)
    ]
    engine = FactorEngine([
        BasicMomentumEngineFactor(window=2),
        BasicVolatilityEngineFactor(window=2),
        BasicVolumeEngineFactor(window=2),
    ])

    result = engine.calculate_one(symbol, bars)

    assert result.scalar_map(symbol).keys() == {
        "momentum_2",
        "volatility_2",
        "volume_2",
    }
    assert all(
        isinstance(value.value, float)
        for value in result.for_symbol(symbol).values()
    )

    latest_bar = SimpleNamespace(
        symbol=symbol,
        datetime=datetime(2026, 7, 20, 10),
        close=13.0,
    )
    sample = FactorSampleAssembler().build_sample(latest_bar, result)

    assert sample is not None
    assert sample.symbol == symbol
    assert sample.volume_ratio == result.get(symbol, "volume_2").value


def test_strategy_consumes_normalized_factor_values() -> None:
    symbol = "000001.SSE"
    strategy = SimpleFactorBuyStrategy(
        strategy_engine=None,
        strategy_name="simple",
        symbols=[symbol],
        setting={"min_return": 0.02},
    )
    context = StrategyContext()
    strategy.on_init(context)
    result = FactorBatchResult(values=[
        FactorValue(
            symbol=symbol,
            factor_name="momentum_20",
            value=0.03,
            fields={"ret_n": 0.03, "trend": "UP"},
        ),
        FactorValue(
            symbol=symbol,
            factor_name="volatility_20",
            value=0.01,
        ),
        FactorValue(
            symbol=symbol,
            factor_name="volume_20",
            value=1.2,
            fields={"price_volume_signal": "NORMAL"},
        ),
    ])

    outputs = strategy.on_factor(
        context,
        SimpleNamespace(symbol=symbol, close=10.0),
        result,
    )

    assert len(outputs) == 1
    assert outputs[0].direction == SignalDirection.LONG

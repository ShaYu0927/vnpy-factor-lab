import polars as pl
import pytest

from vnpy.alpha import AlphaEngine
from examples.alpha101_debug import demo_frame


def test_alpha101_panel_alignment_and_samples():
    frame = demo_frame()
    engine = AlphaEngine([])
    result = engine.calculate_alpha101(frame.reverse(), [33, 101])
    assert result.columns == ["vt_symbol", "datetime", "alpha033", "alpha101"]
    assert result.height == frame.height
    joined = result.join(frame, on=["datetime", "vt_symbol"])
    expected = (joined["close"] - joined["open"]) / (joined["high"] - joined["low"] + 0.001)
    assert joined["alpha101"].to_list() == pytest.approx(expected.to_list())
    # Cross-sectional ranks must use all three symbols, even with reversed input.
    for group in result.partition_by("datetime"):
        assert sorted(group["alpha033"].to_list()) == pytest.approx([1/3, 2/3, 1])
    samples = engine.calculate_alpha101_latest(frame, [101])
    assert len(samples) == 3
    assert set(samples[0].features) == {"alpha101"}
    assert engine.to_observations(samples)[0].features == samples[0].features


def test_all_factors_and_missing_market_cap():
    engine = AlphaEngine([])
    frame = demo_frame()
    result = engine.calculate_alpha101(frame)
    assert result.shape == (960, 103)
    missing = engine.calculate_alpha101(frame.drop("market_cap"), [56])
    assert missing["alpha056"].null_count() == frame.height
    assert engine.calculate_alpha101_latest(frame.drop("market_cap"), [56]) == []


@pytest.mark.parametrize("factors", [[], [1, 1], [0], [102], [1.5]])
def test_invalid_factor_selection(factors):
    with pytest.raises(ValueError):
        AlphaEngine([]).calculate_alpha101(demo_frame(), factors)

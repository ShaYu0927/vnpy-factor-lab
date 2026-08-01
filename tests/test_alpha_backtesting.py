from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.alpha.strategy.template import AlphaStrategy
from vnpy.trader.constant import Exchange, Interval, Status
from vnpy.trader.object import BarData, TradeData


VT_SYMBOL_A = "000001.SSE"
VT_SYMBOL_B = "000002.SSE"


def make_bar(symbol: str, dt: datetime, price: float = 10) -> BarData:
    return BarData(
        gateway_name="TEST",
        symbol=symbol,
        exchange=Exchange.SSE,
        datetime=dt,
        interval=Interval.DAILY,
        open_price=price,
        high_price=price + 1,
        low_price=price - 1,
        close_price=price,
    )


class FakeLab:
    def __init__(self, bars: dict[str, list[BarData]]) -> None:
        self.bars = bars

    def load_contract_setttings(self) -> dict:
        return {
            vt_symbol: {
                "long_rate": 0.001,
                "short_rate": 0.001,
                "size": 1,
                "pricetick": 0.01,
            }
            for vt_symbol in self.bars
        }

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        return self.bars[vt_symbol]


class BuyOnceStrategy(AlphaStrategy):
    def on_init(self) -> None:
        self.has_ordered = False

    def on_bars(self, bars: dict[str, BarData]) -> None:
        if not self.has_ordered and VT_SYMBOL_A in bars:
            self.buy(VT_SYMBOL_A, 20, 1)
            self.has_ordered = True

    def on_trade(self, trade: TradeData) -> None:
        pass


class NoTradeStrategy(AlphaStrategy):
    def on_init(self) -> None:
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        pass


def make_engine(
    strategy_class: type[AlphaStrategy],
    bars: dict[str, list[BarData]],
) -> BacktestingEngine:
    start = min(bar.datetime for values in bars.values() for bar in values)
    end = max(bar.datetime for values in bars.values() for bar in values) + timedelta(days=1)
    engine = BacktestingEngine(FakeLab(bars))  # type: ignore[arg-type]
    engine.set_parameters(list(bars), Interval.DAILY, start, end)
    engine.add_strategy(strategy_class, {}, pl.DataFrame())
    engine.load_data()
    return engine


def test_missing_bar_is_used_for_valuation_but_not_order_matching() -> None:
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    day3 = datetime(2024, 1, 4)
    engine = make_engine(
        BuyOnceStrategy,
        {
            VT_SYMBOL_A: [make_bar("000001", day1), make_bar("000001", day3)],
            VT_SYMBOL_B: [make_bar("000002", day2)],
        },
    )

    engine.strategy.on_init()
    engine.new_bars(day1)
    order = next(iter(engine.active_limit_orders.values()))

    engine.new_bars(day2)
    assert not engine.trades
    assert order.status == Status.NOTTRADED
    assert engine.bars[VT_SYMBOL_A].datetime == day2

    engine.new_bars(day3)
    assert len(engine.trades) == 1
    assert order.status == Status.ALLTRADED


def test_result_calculation_is_idempotent() -> None:
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    engine = make_engine(
        BuyOnceStrategy,
        {VT_SYMBOL_A: [make_bar("000001", day1), make_bar("000001", day2, 11)]},
    )
    engine.run_backtesting()

    first = engine.calculate_result()
    second = engine.calculate_result()

    assert first is not None
    assert second is not None
    assert first.equals(second)


def test_no_trade_backtest_returns_zero_pnl_results() -> None:
    day1 = datetime(2024, 1, 2)
    engine = make_engine(
        NoTradeStrategy,
        {VT_SYMBOL_A: [make_bar("000001", day1)]},
    )
    engine.run_backtesting()

    result = engine.calculate_result()

    assert result is not None
    assert result["trade_count"].sum() == 0
    assert result["net_pnl"].sum() == 0

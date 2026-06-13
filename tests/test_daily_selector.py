from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from vnpy.selection.daily_selector import (
    DailySelectionConfig,
    DailyStockSelector,
)


class EmptyProvider:
    def get_universe(self, query_date: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return pd.DataFrame()


class InMemoryProvider:
    def __init__(self) -> None:
        self.frames = {
            "sh.600001": make_bars(0.003, 80_000_000, 10.0, 1.0),
            "sh.600002": make_bars(0.001, 40_000_000, 20.0, 2.0),
            "sh.600003": make_bars(-0.001, 20_000_000, 30.0, 3.0),
        }

    def get_universe(self, query_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": list(self.frames),
                "code_name": ["one", "two", "three"],
                "industry": ["bank", "bank", "industry"],
            }
        )

    def get_daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self.frames[code]


def make_bars(
    daily_return: float,
    amount: float,
    pe_ttm: float,
    pb_mrq: float,
    periods: int = 100,
) -> pd.DataFrame:
    closes = 10.0 * np.power(1.0 + daily_return, np.arange(periods))
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=periods, freq="B"),
            "close": closes,
            "amount": amount,
            "tradestatus": "1",
            "isST": "0",
            "peTTM": pe_ttm,
            "pbMRQ": pb_mrq,
        }
    )


class DailyStockSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        config = DailySelectionConfig(
            top_n=2,
            min_average_amount=1_000_000,
        )
        self.selector = DailyStockSelector(EmptyProvider(), config)

    def test_calculate_metrics(self) -> None:
        metrics = self.selector.calculate_metrics(
            "sh.600000",
            make_bars(0.002, 50_000_000, 10.0, 1.2),
        )

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics["momentum_20"], 0)
        self.assertGreater(metrics["momentum_60"], metrics["momentum_20"])
        self.assertAlmostEqual(metrics["average_amount_20"], 50_000_000)
        self.assertEqual(metrics["trade_date"], "2026-05-20")

    def test_filters_st_and_low_liquidity(self) -> None:
        st_bars = make_bars(0.001, 50_000_000, 10.0, 1.2)
        st_bars.loc[st_bars.index[-1], "isST"] = "1"
        self.assertIsNone(self.selector.calculate_metrics("sh.600001", st_bars))

        low_liquidity = make_bars(0.001, 100_000, 10.0, 1.2)
        self.assertIsNone(
            self.selector.calculate_metrics("sh.600002", low_liquidity)
        )

    def test_rank_returns_top_n(self) -> None:
        rows = []
        specs = [
            ("sh.600001", 0.0030, 80_000_000, 10.0, 1.0),
            ("sh.600002", 0.0010, 40_000_000, 20.0, 2.0),
            ("sh.600003", -0.0010, 20_000_000, 30.0, 3.0),
        ]
        for code, daily_return, amount, pe_ttm, pb_mrq in specs:
            metrics = self.selector.calculate_metrics(
                code,
                make_bars(daily_return, amount, pe_ttm, pb_mrq),
            )
            metrics["name"] = code
            metrics["industry"] = "test"
            rows.append(metrics)

        ranked = self.selector.rank(pd.DataFrame(rows))

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["code"], "sh.600001")
        self.assertEqual(ranked["rank"].tolist(), [1, 2])
        self.assertTrue(ranked["total_score"].is_monotonic_decreasing)

    def test_select_runs_end_to_end_with_provider(self) -> None:
        selector = DailyStockSelector(InMemoryProvider(), self.selector.config)
        result = selector.select("2026-05-20")

        self.assertEqual(result["code"].tolist(), ["sh.600001", "sh.600002"])
        self.assertEqual(result["name"].tolist(), ["one", "two"])


if __name__ == "__main__":
    unittest.main()

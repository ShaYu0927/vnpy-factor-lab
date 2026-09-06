from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np
import pandas as pd

from vnpy.alpha.logger import logger

class Alpha101:
    """Calculate the 101 formulaic alphas on long-form daily market data."""

    REQUIRED = {"open", "high", "low", "close", "volume"}

    def __init__(self, data: pd.DataFrame, *, eps: float = 1e-12) -> None:
        self.eps = float(eps)
        self.data = self._prepare(data)
        self._adv_cache: dict[int, pd.Series] = {}
        logger.info("Alpha101 input: rows=%d symbols=%d dates=%d", len(self.data),
                    self.data.index.get_level_values("symbol").nunique(),
                    self.data.index.get_level_values("date").nunique())
        fallback = sorted({"vwap", "market_cap", "industry", "sector", "subindustry"} - set(data.columns))
        if fallback:
            logger.warning("Alpha101 fallback fields=%s (vwap=typical price, market_cap=NaN, industry=ALL)", fallback)
        
    @classmethod
    def _prepare(cls, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if not isinstance(data.index, pd.MultiIndex):
            if not {"symbol", "date"}.issubset(data.columns):
                raise ValueError("data must contain date and symbol columns")
            data["date"] = pd.to_datetime(data["date"])
            data = data.set_index(["symbol", "date"])
        else:
            if not {"symbol", "date"}.issubset(data.index.names):
                raise ValueError("MultiIndex levels must be named symbol and date")
            if list(data.index.names) != ["symbol", "date"]:
                data = data.reorder_levels(["symbol", "date"])
                
        if data.index.has_duplicates:
            raise ValueError("duplicate (symbol, date) rows are not allowed")
        missing = cls.REQUIRED - set(data.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        
        data = data.sort_index()
        
        for column in cls.REQUIRED:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            
        if "return" not in data:
            data["return"] = data.groupby(level="symbol")["close"].pct_change(fill_method=None)
        
        if "vwap" not in data:
            data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3.0
        if "market_cap" not in data:
            data["market_cap"] = np.nan
        for column in ("return", "vwap", "market_cap"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
            
        if "industry" not in data:
            data["industry"] = "ALL"
        if "sector" not in data:
            data["sector"] = data["industry"]
        if "subindustry" not in data:
            data["subindustry"] = data["industry"]
        return data
    
    def _rank(self, x: pd.Series) -> pd.Series:
        return x.groupby(level="date").rank(method="average", pct=True)
    
    def _delay(self, x: pd.Series, d: int) -> pd.Series:
        return x.groupby(level="symbol", group_keys=False).shift(int(d))

    def _delta(self, x: pd.Series, d: int = 1) -> pd.Series:
        return x.groupby(level="symbol", group_keys=False).diff(int(d))
    
    def _rolling(self, x: pd.Series, d: int, op: str) -> pd.Series:
        grouped = x.groupby(level="symbol", group_keys=False)
        result = getattr(grouped.rolling(int(d), min_periods=int(d)), op)()
        return result.droplevel(0).reindex(self.data.index)
    
    def _sum(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling(x, d, "sum")
    
    def _mean(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling(x, d, "mean")
    
    def _std(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling(x, d, "std")

    def _ts_min(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling(x, d, "min")
    
    def _ts_max(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling(x, d, "max")

    def _corr(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        pair = pd.concat({"x": x, "y": y}, axis=1)
        
        def rolling_corr(group: pd.DataFrame) -> pd.Series:
            return group["x"].rolling(int(d), min_periods=int(d)).corr(group["y"])

        out = pair.groupby(level="symbol", group_keys=False).apply(rolling_corr)
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
            out = out.droplevel(0)
        return out.reindex(self.data.index).replace([np.inf, -np.inf], np.nan)
    
    def _cov(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        pair = pd.concat({"x": x, "y": y}, axis=1)

        def rolling_cov(group: pd.DataFrame) -> pd.Series:
            return group["x"].rolling(int(d), min_periods=int(d)).cov(group["y"])

        out = pair.groupby(level="symbol", group_keys=False).apply(rolling_cov)
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
            out = out.droplevel(0)
        return out.reindex(self.data.index)
    
    def _ts_rank(self, x: pd.Series, d: int) -> pd.Series:
        out = (
            x.groupby(level="symbol")
            .rolling(int(d), min_periods=int(d))
            .rank(method="average", pct=True)
            .droplevel(0)
        )
        return out.reindex(self.data.index)
    
    
    def _ts_argmin(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling_apply(x, d, lambda a: float(np.argmin(a) + 1))
    
    def _ts_argmax(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling_apply(x, d, lambda a: float(np.argmax(a) + 1))
    
    def _product(self, x: pd.Series, d: int) -> pd.Series:
        return self._rolling_apply(x, d, np.prod)
    
    def _rolling_apply(self, x: pd.Series, d: int, func) -> pd.Series:
        out = (
            x.groupby(level="symbol")
            .rolling(int(d), min_periods=int(d))
            .apply(func, raw=True)
            .droplevel(0)
        )
        return out.reindex(self.data.index)
    
    def _decay_linear(self, x: pd.Series, d: int) -> pd.Series:
        window = int(d)
        weights = np.arange(1.0, window + 1.0)
        weights /= weights.sum()
        return self._rolling_apply(x, window, lambda a: float(np.dot(a, weights)))

    def _scale(self, x: pd.Series, a: float = 1.0) -> pd.Series:
        def scale_group(group: pd.Series) -> pd.Series:
            denom = group.abs().sum(min_count=1)
            return group * a / denom if denom and np.isfinite(denom) else group * np.nan

        return x.groupby(level="date", group_keys=False).apply(scale_group)
    
    def _neutralize(self, x: pd.Series, level: str) -> pd.Series:
        groups = self.data[level]
        dates = self.data.index.get_level_values("date")
        means = x.groupby([dates, groups], dropna=False).transform("mean")
        return x - means
    
    def _adv(self, d: int) -> pd.Series:
        window = int(d)
        if window not in self._adv_cache:
            self._adv_cache[window] = self._mean(self.data["volume"], window)
        return self._adv_cache[window]
    
    @staticmethod
    def _signed_power(x: pd.Series, exponent: pd.Series | float) -> pd.Series:
        return np.sign(x) * np.power(np.abs(x), exponent)
    
    @staticmethod
    def _min2(x: pd.Series, y: pd.Series) -> pd.Series:
        return pd.concat([x, y], axis=1).min(axis=1, skipna=False)
    
    @staticmethod
    def _max2(x: pd.Series, y: pd.Series) -> pd.Series:
        return pd.concat([x, y], axis=1).max(axis=1, skipna=False)
    
    def compute(self, number: int) -> pd.Series:
        if not 1 <= int(number) <= 101:
            raise ValueError("factor number must be in 1..101")
        method = getattr(self, f"alpha{int(number):03d}")
        result = method().reindex(self.data.index)
        result.name = f"alpha{int(number):03d}"
        return result.astype(float).replace([np.inf, -np.inf], np.nan)
    
    
    def compute_all(self, factors: Iterable[int] | None = None, *, at=None) -> pd.DataFrame:
        """Optionally retain only one date, releasing each historical factor column."""
        numbers = list(range(1, 102) if factors is None else factors)
        if not numbers:
            raise ValueError("factors must not be empty")
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 1 <= n <= 101 for n in numbers):
            raise ValueError("factor numbers must be integers in 1..101")
        if len(numbers) != len(set(numbers)):
            raise ValueError("factor numbers must be unique")
        started = perf_counter()
        mask = None if at is None else self.data.index.get_level_values("date") == pd.Timestamp(at)
        columns = []
        for position, number in enumerate(numbers, 1):
            tick = perf_counter()
            try:
                values = self.compute(number)
                if mask is not None:
                    values = values.loc[mask].copy()
                    # Bound retained ADV arrays as well as completed factor columns.
                    self._adv_cache.clear()
            except Exception:
                logger.exception("Alpha101 failed: alpha%03d", number)
                raise
            columns.append(values)
            logger.info("Alpha101 [%d/%d] %s valid=%d/%d elapsed=%.3fs",
                        position, len(numbers), values.name, values.notna().sum(),
                        len(values), perf_counter() - tick)
        result = pd.concat(columns, axis=1)
        logger.info("Alpha101 complete: shape=%s elapsed=%.3fs", result.shape, perf_counter() - started)
        logger.debug("Alpha101 result preview:\n%s", result.tail(5).to_string())
        return result
    
    def alpha001(self) -> pd.Series:
        r, c = self.data["return"], self.data["close"]
        inner = self._std(r, 20).where(r < 0, c)
        return self._rank(self._ts_argmax(self._signed_power(inner, 2.0), 5)) - 0.5

    
    def alpha002(self) -> pd.Series:
        d = self.data
        x = self._rank(self._delta(np.log(d["volume"].where(d["volume"] > 0)), 2))
        y = self._rank((d["close"] - d["open"]) / (d["open"] + self.eps))
        return -self._corr(x, y, 6)
    
    def alpha003(self) -> pd.Series:
        return -self._corr(self._rank(self.data["open"]), self._rank(self.data["volume"]), 10)
    
    def alpha004(self) -> pd.Series:
        return -self._ts_rank(self._rank(self.data["low"]), 9)

    def alpha005(self) -> pd.Series:
        d = self.data
        left = self._rank(d["open"] - self._mean(d["vwap"], 10))
        right = -self._rank(d["close"] - d["vwap"]).abs()
        return left * right
    
    def alpha006(self) -> pd.Series:
        return -self._corr(self.data["open"], self.data["volume"], 10)

    def alpha007(self) -> pd.Series:
        d = self.data
        dc = self._delta(d["close"], 7)
        signal = -self._ts_rank(dc.abs(), 60) * np.sign(dc)
        return signal.where(self._adv(20) < d["volume"], -1.0)
    
    def alpha008(self) -> pd.Series:
        d = self.data
        inner = self._sum(d["open"], 5) * self._sum(d["return"], 5)
        return -self._rank(inner - self._delay(inner, 10))
    
    def alpha009(self) -> pd.Series:
        dc = self._delta(self.data["close"], 1)
        same_sign = (self._ts_min(dc, 5) > 0) | (self._ts_max(dc, 5) < 0)
        return dc.where(same_sign, -dc)
    
    def alpha010(self) -> pd.Series:
        dc = self._delta(self.data["close"], 1)
        same_sign = (self._ts_min(dc, 4) > 0) | (self._ts_max(dc, 4) < 0)
        return self._rank(dc.where(same_sign, -dc))
    
    def alpha011(self) -> pd.Series:
        d = self.data
        spread = d["vwap"] - d["close"]
        return (
            self._rank(self._ts_max(spread, 3)) + self._rank(self._ts_min(spread, 3))
        ) * self._rank(self._delta(d["volume"], 3))
        
    def alpha012(self) -> pd.Series:
        d = self.data
        return np.sign(self._delta(d["volume"], 1)) * -self._delta(d["close"], 1)
    
    def alpha013(self) -> pd.Series:
        d = self.data
        return -self._rank(self._cov(self._rank(d["close"]), self._rank(d["volume"]), 5))

    def alpha014(self) -> pd.Series:
        d = self.data
        return -self._rank(self._delta(d["return"], 3)) * self._corr(d["open"], d["volume"], 10)
    
    def alpha015(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._rank(d["high"]), self._rank(d["volume"]), 3)
        return -self._sum(self._rank(corr), 3)
    
    def alpha016(self) -> pd.Series:
        d = self.data
        return -self._rank(self._cov(self._rank(d["high"]), self._rank(d["volume"]), 5))

    def alpha017(self) -> pd.Series:
        d = self.data
        return (
            -self._rank(self._ts_rank(d["close"], 10))
            * self._rank(self._delta(self._delta(d["close"], 1), 1))
            * self._rank(self._ts_rank(d["volume"] / (self._adv(20) + self.eps), 5))
        )
        
    def alpha018(self) -> pd.Series:
        d = self.data
        inner = self._std((d["close"] - d["open"]).abs(), 5)
        inner = inner + (d["close"] - d["open"]) + self._corr(d["close"], d["open"], 10)
        return -self._rank(inner)
    
    def alpha019(self) -> pd.Series:
        d = self.data
        left = -np.sign(d["close"] - self._delay(d["close"], 7) + self._delta(d["close"], 7))
        return left * (1 + self._rank(1 + self._sum(d["return"], 250)))
    
    def alpha020(self) -> pd.Series:
        d = self.data
        return (
            -self._rank(d["open"] - self._delay(d["high"], 1))
            * self._rank(d["open"] - self._delay(d["close"], 1))
            * self._rank(d["open"] - self._delay(d["low"], 1))
        )
        
    def alpha021(self) -> pd.Series:
        d = self.data
        mean8, mean2, std8 = self._mean(d["close"], 8), self._mean(d["close"], 2), self._std(d["close"], 8)
        ratio = d["volume"] / (self._adv(20) + self.eps)
        out = pd.Series(np.where(mean8 + std8 < mean2, -1.0,
                       np.where(mean2 < mean8 - std8, 1.0,
                       np.where(ratio >= 1.0, 1.0, -1.0))), index=d.index)
        return out.where(mean8.notna() & mean2.notna() & std8.notna() & self._adv(20).notna())

    def alpha022(self) -> pd.Series:
        d = self.data
        return -self._delta(self._corr(d["high"], d["volume"], 5), 5) * self._rank(self._std(d["close"], 20))
    
    def alpha023(self) -> pd.Series:
        d = self.data
        mean_high = self._mean(d["high"], 20)
        return (-self._delta(d["high"], 2)).where(mean_high < d["high"], 0.0).where(mean_high.notna())
    
    def alpha024(self) -> pd.Series:
        d = self.data
        trend = self._delta(self._mean(d["close"], 100), 100) / (self._delay(d["close"], 100) + self.eps)
        slow = -(d["close"] - self._ts_min(d["close"], 100))
        fast = -self._delta(d["close"], 3)
        return slow.where(trend <= 0.05, fast).where(trend.notna())
    
    def alpha025(self) -> pd.Series:
        d = self.data
        return self._rank(-d["return"] * self._adv(20) * d["vwap"] * (d["high"] - d["close"]))
    
    def alpha026(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._ts_rank(d["volume"], 5), self._ts_rank(d["high"], 5), 5)
        return -self._ts_max(corr, 3)
    
    
    def alpha027(self) -> pd.Series:
        d = self.data
        inner = self._rank(self._mean(self._corr(self._rank(d["volume"]), self._rank(d["vwap"]), 6), 2))
        return pd.Series(np.where(inner > 0.5, -1.0, 1.0), index=d.index).where(inner.notna())
    
    def alpha028(self) -> pd.Series:
        d = self.data
        return self._scale(self._corr(self._adv(20), d["low"], 5) + (d["high"] + d["low"]) / 2 - d["close"])
    
    
    def alpha029(self) -> pd.Series:
        d = self.data
        inner = -self._rank(self._delta(d["close"] - 1.0, 5))
        inner = self._rank(self._rank(inner))
        inner = self._ts_min(inner, 2)
        inner = self._sum(inner, 1)
        inner = np.log(self._scale(inner).where(self._scale(inner) > 0))
        left = self._ts_min(self._rank(self._rank(inner)), 5)
        right = self._ts_rank(self._delay(-d["return"], 6), 5)
        return left + right
    
    def alpha030(self) -> pd.Series:
        d = self.data
        c = d["close"]
        inner = np.sign(c - self._delay(c, 1))
        inner += np.sign(self._delay(c, 1) - self._delay(c, 2))
        inner += np.sign(self._delay(c, 2) - self._delay(c, 3))
        return (1.0 - self._rank(inner)) * self._sum(d["volume"], 5) / (self._sum(d["volume"], 20) + self.eps)
    
    def alpha031(self) -> pd.Series:
        d = self.data
        p0 = self._rank(self._rank(self._rank(self._decay_linear(-self._rank(self._rank(self._delta(d["close"], 10))), 10))))
        p1 = self._rank(-self._delta(d["close"], 3))
        p2 = np.sign(self._scale(self._corr(self._adv(20), d["low"], 12)))
        return p0 + p1 + p2
    
    def alpha032(self) -> pd.Series:
        d = self.data
        left = self._scale(self._mean(d["close"], 7) - d["close"])
        right = 20.0 * self._scale(self._corr(d["vwap"], self._delay(d["close"], 5), 230))
        return left + right
    
    def alpha033(self) -> pd.Series:
        d = self.data
        return self._rank(-(1.0 - d["open"] / (d["close"] + self.eps)))
    
    def alpha034(self) -> pd.Series:
        d = self.data
        vol_ratio = self._std(d["return"], 2) / (self._std(d["return"], 5) + self.eps)
        return self._rank((1.0 - self._rank(vol_ratio)) + (1.0 - self._rank(self._delta(d["close"], 1))))
    
    def alpha035(self) -> pd.Series:
        d = self.data
        return self._ts_rank(d["volume"], 32) * (1.0 - self._ts_rank(d["close"] + d["high"] - d["low"], 16)) * (1.0 - self._ts_rank(d["return"], 32))
    
    def alpha036(self) -> pd.Series:
        d = self.data
        p0 = 2.21 * self._rank(self._corr(d["close"] - d["open"], self._delay(d["volume"], 1), 15))
        p1 = 0.7 * self._rank(d["open"] - d["close"])
        p2 = 0.73 * self._rank(self._ts_rank(self._delay(-d["return"], 6), 5))
        p3 = self._rank(self._corr(d["vwap"], self._adv(20), 6).abs())
        p4 = 0.6 * self._rank((self._mean(d["close"], 200) - d["open"]) * (d["close"] - d["open"]))
        return p0 + p1 + p2 + p3 + p4
    
    def alpha037(self) -> pd.Series:
        d = self.data
        return self._rank(self._corr(self._delay(d["open"] - d["close"], 1), d["close"], 200)) + self._rank(d["open"] - d["close"])
    
    def alpha038(self) -> pd.Series:
        d = self.data
        return -self._rank(self._ts_rank(d["close"], 10)) * self._rank(d["close"] / (d["open"] + self.eps))
    
    def alpha039(self) -> pd.Series:
        d = self.data
        left = -self._rank(self._delta(d["close"], 7) * (1.0 - self._rank(self._decay_linear(d["volume"] / (self._adv(20) + self.eps), 9))))
        return left * (1.0 + self._rank(self._sum(d["return"], 250)))
    
    def alpha040(self) -> pd.Series:
        d = self.data
        return -self._rank(self._std(d["high"], 10)) * self._corr(d["high"], d["volume"], 10)

    def alpha041(self) -> pd.Series:
        d = self.data
        return np.sqrt(d["high"] * d["low"]) - d["vwap"]
    
    def alpha042(self) -> pd.Series:
        d = self.data
        return self._rank(d["vwap"] - d["close"]) / (self._rank(d["vwap"] + d["close"]) + self.eps)
    
    def alpha043(self) -> pd.Series:
        d = self.data
        return self._ts_rank(d["volume"] / (self._adv(20) + self.eps), 20) * self._ts_rank(-self._delta(d["close"], 7), 8)
    
    def alpha044(self) -> pd.Series:
        d = self.data
        return -self._corr(d["high"], self._rank(d["volume"]), 5)
    
    def alpha045(self) -> pd.Series:
        d = self.data
        p0 = self._rank(self._mean(self._delay(d["close"], 5), 20))
        p1 = self._corr(d["close"], d["volume"], 2)
        p2 = self._rank(self._corr(self._sum(d["close"], 5), self._sum(d["close"], 20), 2))
        return -p0 * p1 * p2
    
    def alpha046(self) -> pd.Series:
        c = self.data["close"]
        slope_change = (self._delay(c, 20) - self._delay(c, 10)) / 10.0 - (self._delay(c, 10) - c) / 10.0
        fallback = -self._delta(c, 1)
        out = pd.Series(np.where(slope_change > 0.25, -1.0, np.where(slope_change < 0.0, 1.0, fallback)), index=c.index)
        return out.where(slope_change.notna())
    
    def alpha047(self) -> pd.Series:
        d = self.data
        p0 = self._rank(1.0 / (d["close"] + self.eps)) * d["volume"] / (self._adv(20) + self.eps)
        p1 = d["high"] * self._rank(d["high"] - d["close"]) / (self._mean(d["high"], 5) + self.eps)
        p2 = self._rank(d["vwap"] - self._delay(d["vwap"], 5))
        return p0 * p1 - p2
    
    def alpha048(self) -> pd.Series:
        d = self.data
        dc = self._delta(d["close"], 1)
        numerator = self._neutralize(self._corr(dc, self._delta(self._delay(d["close"], 1), 1), 250) * dc / (d["close"] + self.eps), "subindustry")
        denominator = self._sum((dc / (self._delay(d["close"], 1) + self.eps)) ** 2, 250)
        return numerator / (denominator + self.eps)
    
    def alpha049(self) -> pd.Series:
        c = self.data["close"]
        slope_change = (self._delay(c, 20) - self._delay(c, 10)) / 10.0 - (self._delay(c, 10) - c) / 10.0
        out = pd.Series(np.where(slope_change < -0.1, 1.0, -self._delta(c, 1)), index=c.index)
        return out.where(slope_change.notna())
    
    def alpha050(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._rank(d["volume"]), self._rank(d["vwap"]), 5)
        return -self._ts_max(self._rank(corr), 5)
    
    def alpha051(self) -> pd.Series:
        c = self.data["close"]
        slope_change = (self._delay(c, 20) - self._delay(c, 10)) / 10.0 - (self._delay(c, 10) - c) / 10.0
        out = pd.Series(np.where(slope_change < -0.05, 1.0, -self._delta(c, 1)), index=c.index)
        return out.where(slope_change.notna())
    
    def alpha052(self) -> pd.Series:
        d = self.data
        low5 = self._ts_min(d["low"], 5)
        p0 = -low5 + self._delay(low5, 5)
        p1 = self._rank((self._sum(d["return"], 240) - self._sum(d["return"], 20)) / 220.0)
        return p0 * p1 * self._ts_rank(d["volume"], 5)
    
    def alpha053(self) -> pd.Series:
        d = self.data
        inner = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["close"] - d["low"] + self.eps)
        return -self._delta(inner, 9)

    def alpha054(self) -> pd.Series:
        d = self.data
        return -((d["low"] - d["close"]) * d["open"] ** 5) / ((d["low"] - d["high"]) * d["close"] ** 5 + self.eps)
    
    
    def alpha055(self) -> pd.Series:
        d = self.data
        position = (d["close"] - self._ts_min(d["low"], 12)) / (self._ts_max(d["high"], 12) - self._ts_min(d["low"], 12) + self.eps)
        return -self._corr(self._rank(position), self._rank(d["volume"]), 6)
    
    def alpha056(self) -> pd.Series:
        d = self.data
        left = self._rank(self._sum(d["return"], 10) / (self._sum(self._sum(d["return"], 2), 3) + self.eps))
        return -left * self._rank(d["return"] * d["market_cap"])
    
    def alpha057(self) -> pd.Series:
        d = self.data
        denominator = self._decay_linear(self._rank(self._ts_argmax(d["close"], 30)), 2)
        return -(d["close"] - d["vwap"]) / (denominator + self.eps)
    
    def alpha058(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._neutralize(d["vwap"], "sector"), d["volume"], 4)
        return -self._ts_rank(self._decay_linear(corr, 8), 6)
    
    def alpha059(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._neutralize(d["vwap"], "industry"), d["volume"], 4)
        return -self._ts_rank(self._decay_linear(corr, 16), 8)
    
    def alpha060(self) -> pd.Series:
        d = self.data
        location = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["high"] - d["low"] + self.eps)
        left = 2.0 * self._scale(self._rank(location * d["volume"]))
        right = self._scale(self._rank(self._ts_argmax(d["close"], 10)))
        return -(left - right)

    def alpha061(self) -> pd.Series:
        d = self.data
        left = self._rank(d["vwap"] - self._ts_min(d["vwap"], 16))
        right = self._rank(self._corr(d["vwap"], self._adv(180), 18))
        return (left < right).astype(float).where(left.notna() & right.notna())

    def alpha062(self) -> pd.Series:
        d = self.data
        left = self._rank(self._corr(d["vwap"], self._sum(self._adv(20), 22), 10))
        comparison = (self._rank(d["open"]) + self._rank(d["open"])) < (self._rank((d["high"] + d["low"]) / 2) + self._rank(d["high"]))
        right = self._rank(comparison.astype(float))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha063(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._delta(self._neutralize(d["close"], "industry"), 2), 8))
        blend = d["vwap"] * 0.318108 + d["open"] * (1.0 - 0.318108)
        right = self._rank(self._decay_linear(self._corr(blend, self._sum(self._adv(180), 37), 14), 12))
        return -(left - right)
    
    def alpha064(self) -> pd.Series:
        d = self.data
        blend0 = d["open"] * 0.178404 + d["low"] * (1.0 - 0.178404)
        left = self._rank(self._corr(self._sum(blend0, 13), self._sum(self._adv(120), 13), 17))
        blend1 = ((d["high"] + d["low"]) / 2) * 0.178404 + d["vwap"] * (1.0 - 0.178404)
        right = self._rank(self._delta(blend1, 4))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha065(self) -> pd.Series:
        d = self.data
        blend = d["open"] * 0.00817205 + d["vwap"] * (1.0 - 0.00817205)
        left = self._rank(self._corr(blend, self._sum(self._adv(60), 9), 6))
        right = self._rank(d["open"] - self._ts_min(d["open"], 14))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha066(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._delta(d["vwap"], 4), 7))
        ratio = (d["low"] - d["vwap"]) / (d["open"] - (d["high"] + d["low"]) / 2 + self.eps)
        right = self._ts_rank(self._decay_linear(ratio, 11), 7)
        return -(left + right)
    
    def alpha067(self) -> pd.Series:
        d = self.data
        left = self._rank(d["high"] - self._ts_min(d["high"], 2))
        corr = self._corr(self._neutralize(d["vwap"], "sector"), self._neutralize(self._adv(20), "subindustry"), 6)
        right = self._rank(corr)
        return -np.power(left, right)
    
    def alpha068(self) -> pd.Series:
        d = self.data
        left = self._ts_rank(self._corr(self._rank(d["high"]), self._rank(self._adv(15)), 9), 14)
        blend = d["close"] * 0.518371 + d["low"] * (1.0 - 0.518371)
        right = self._rank(self._delta(blend, 1))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    
    def alpha069(self) -> pd.Series:
        d = self.data
        left = self._rank(self._ts_max(self._delta(self._neutralize(d["vwap"], "industry"), 3), 5))
        blend = d["close"] * 0.490655 + d["vwap"] * (1.0 - 0.490655)
        right = self._ts_rank(self._corr(blend, self._adv(20), 5), 9)
        return -np.power(left, right)
    
    def alpha070(self) -> pd.Series:
        d = self.data
        left = self._rank(self._delta(d["vwap"], 1))
        right = self._ts_rank(self._corr(self._neutralize(d["close"], "industry"), self._adv(50), 18), 18)
        return -np.power(left, right)
    
    def alpha071(self) -> pd.Series:
        d = self.data
        left = self._ts_rank(self._decay_linear(self._corr(self._ts_rank(d["close"], 3), self._ts_rank(self._adv(180), 12), 18), 4), 16)
        inner = self._rank(d["low"] + d["open"] - 2.0 * d["vwap"])
        right = self._ts_rank(self._decay_linear(inner ** 2, 16), 4)
        return self._max2(left, right)
    
    def alpha072(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._corr((d["high"] + d["low"]) / 2, self._adv(40), 9), 10))
        corr = self._corr(self._ts_rank(d["vwap"], 4), self._ts_rank(d["volume"], 19), 7)
        right = self._rank(self._decay_linear(corr, 3))
        return left / (right + self.eps)
    
    def alpha073(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._delta(d["vwap"], 5), 3))
        blend = d["open"] * 0.147155 + d["low"] * (1.0 - 0.147155)
        right = self._ts_rank(self._decay_linear(-self._delta(blend, 2) / (blend + self.eps), 3), 17)
        return -self._max2(left, right)
    
    def alpha074(self) -> pd.Series:
        d = self.data
        left = self._rank(self._corr(d["close"], self._sum(self._adv(30), 37), 15))
        blend = d["high"] * 0.0261661 + d["vwap"] * (1.0 - 0.0261661)
        right = self._rank(self._corr(self._rank(blend), self._rank(d["volume"]), 11))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha075(self) -> pd.Series:
        d = self.data
        left = self._rank(self._corr(d["vwap"], d["volume"], 4))
        right = self._rank(self._corr(self._rank(d["low"]), self._rank(self._adv(50)), 12))
        return (left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha076(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._delta(d["vwap"], 1), 12))
        corr = self._corr(self._neutralize(d["low"], "sector"), self._adv(81), 8)
        right = self._ts_rank(self._decay_linear(self._ts_rank(corr, 20), 17), 19)
        return -self._max2(left, right)
    
    def alpha077(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear((d["high"] + d["low"]) / 2 - d["vwap"], 20))
        right = self._rank(self._decay_linear(self._corr((d["high"] + d["low"]) / 2, self._adv(40), 3), 6))
        return self._min2(left, right)
    
    def alpha078(self) -> pd.Series:
        d = self.data
        blend = d["low"] * 0.352233 + d["vwap"] * (1.0 - 0.352233)
        left = self._rank(self._corr(self._sum(blend, 20), self._sum(self._adv(40), 20), 7))
        right = self._rank(self._corr(self._rank(d["vwap"]), self._rank(d["volume"]), 6))
        return np.power(left, right)
    
    def alpha079(self) -> pd.Series:
        d = self.data
        blend = d["close"] * 0.60733 + d["open"] * (1.0 - 0.60733)
        left = self._rank(self._delta(self._neutralize(blend, "sector"), 1))
        right = self._rank(self._corr(self._ts_rank(d["vwap"], 4), self._ts_rank(self._adv(150), 9), 15))
        return (left < right).astype(float).where(left.notna() & right.notna())
    
    def alpha080(self) -> pd.Series:
        d = self.data
        blend = d["open"] * 0.868128 + d["high"] * (1.0 - 0.868128)
        left = self._rank(np.sign(self._delta(self._neutralize(blend, "industry"), 4)))
        right = self._ts_rank(self._corr(d["high"], self._adv(10), 5), 6)
        return -np.power(left, right)
    
    def alpha081(self) -> pd.Series:
        d = self.data
        corr0 = self._corr(d["vwap"], self._sum(self._adv(10), 50), 8)
        inner = self._rank(np.power(self._rank(corr0), 4.0))
        product15 = self._product(inner, 15)
        left = self._rank(np.log(product15.where(product15 > 0)))
        right = self._rank(self._corr(self._rank(d["vwap"]), self._rank(d["volume"]), 5))
        return -(left < right).astype(float).where(left.notna() & right.notna())


    def alpha082(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._delta(d["open"], 1), 15))
        corr = self._corr(self._neutralize(d["volume"], "sector"), d["open"], 17)
        right = self._ts_rank(self._decay_linear(corr, 7), 13)
        return -self._min2(left, right)
    
    def alpha083(self) -> pd.Series:
        d = self.data
        inner = (d["high"] - d["low"]) / (self._mean(d["close"], 5) + self.eps)
        numerator = self._rank(self._delay(inner, 2)) * self._rank(self._rank(d["volume"]))
        denominator = inner / (d["vwap"] - d["close"] + self.eps)
        return numerator / (denominator + self.eps)
    
    def alpha084(self) -> pd.Series:
        d = self.data
        base = self._ts_rank(d["vwap"] - self._ts_max(d["vwap"], 15), 21)
        return self._signed_power(base, self._delta(d["close"], 5))

    def alpha085(self) -> pd.Series:
        d = self.data
        blend = d["high"] * 0.876703 + d["close"] * (1.0 - 0.876703)
        left = self._rank(self._corr(blend, self._adv(30), 10))
        right_corr = self._corr(self._ts_rank((d["high"] + d["low"]) / 2, 4), self._ts_rank(d["volume"], 10), 7)
        return np.power(left, self._rank(right_corr))
    
    def alpha086(self) -> pd.Series:
        d = self.data
        left = self._ts_rank(self._corr(d["close"], self._sum(self._adv(20), 15), 6), 20)
        right = self._rank(d["close"] - d["vwap"])
        return -(left < right).astype(float).where(left.notna() & right.notna())

    def alpha087(self) -> pd.Series:
        d = self.data
        blend = d["close"] * 0.369701 + d["vwap"] * (1.0 - 0.369701)
        left = self._rank(self._decay_linear(self._delta(blend, 2), 3))
        corr = self._corr(self._neutralize(self._adv(81), "industry"), d["close"], 13).abs()
        right = self._ts_rank(self._decay_linear(corr, 5), 14)
        return -self._max2(left, right)
    
    
    def alpha088(self) -> pd.Series:
        d = self.data
        spread = self._rank(d["open"]) + self._rank(d["low"]) - self._rank(d["high"]) - self._rank(d["close"])
        left = self._rank(self._decay_linear(spread, 8))
        corr = self._corr(self._ts_rank(d["close"], 8), self._ts_rank(self._adv(60), 21), 8)
        right = self._ts_rank(self._decay_linear(corr, 7), 3)
        return self._min2(left, right)
    
    def alpha089(self) -> pd.Series:
        d = self.data
        left = self._ts_rank(self._decay_linear(self._corr(d["low"], self._adv(10), 7), 6), 4)
        right = self._ts_rank(self._decay_linear(self._delta(self._neutralize(d["vwap"], "industry"), 3), 10), 15)
        return left - right
    
    def alpha090(self) -> pd.Series:
        d = self.data
        left = self._rank(d["close"] - self._ts_max(d["close"], 5))
        right = self._ts_rank(self._corr(self._neutralize(self._adv(40), "subindustry"), d["low"], 5), 3)
        return -np.power(left, right)
    
    def alpha091(self) -> pd.Series:
        d = self.data
        corr0 = self._corr(self._neutralize(d["close"], "industry"), d["volume"], 10)
        left = self._ts_rank(self._decay_linear(self._decay_linear(corr0, 16), 4), 5)
        right = self._rank(self._decay_linear(self._corr(d["vwap"], self._adv(30), 4), 3))
        return -(left - right)

    def alpha092(self) -> pd.Series:
        d = self.data
        condition = (((d["high"] + d["low"]) / 2 + d["close"]) < (d["low"] + d["open"])).astype(float)
        left = self._ts_rank(self._decay_linear(condition, 15), 19)
        corr = self._corr(self._rank(d["low"]), self._rank(self._adv(30)), 8)
        right = self._ts_rank(self._decay_linear(corr, 7), 7)
        return self._min2(left, right)
    
    def alpha093(self) -> pd.Series:
        d = self.data
        corr = self._corr(self._neutralize(d["vwap"], "industry"), self._adv(81), 17)
        left = self._ts_rank(self._decay_linear(corr, 20), 8)
        blend = d["close"] * 0.524434 + d["vwap"] * (1.0 - 0.524434)
        right = self._rank(self._decay_linear(self._delta(blend, 3), 16))
        return left / (right + self.eps)
    
    def alpha094(self) -> pd.Series:
        d = self.data
        left = self._rank(d["vwap"] - self._ts_min(d["vwap"], 12))
        corr = self._corr(self._ts_rank(d["vwap"], 20), self._ts_rank(self._adv(60), 4), 18)
        right = self._ts_rank(corr, 3)
        return -np.power(left, right)
    
    def alpha095(self) -> pd.Series:
        d = self.data
        left = self._rank(d["open"] - self._ts_min(d["open"], 12))
        corr = self._corr(self._sum((d["high"] + d["low"]) / 2, 19), self._sum(self._adv(40), 19), 13)
        right = self._ts_rank(np.power(self._rank(corr), 5.0), 12)
        return (left < right).astype(float).where(left.notna() & right.notna())
    
    
    def alpha096(self) -> pd.Series:
        d = self.data
        left_corr = self._corr(self._rank(d["vwap"]), self._rank(d["volume"]), 4)
        left = self._ts_rank(self._decay_linear(left_corr, 4), 8)
        right_corr = self._corr(self._ts_rank(d["close"], 7), self._ts_rank(self._adv(60), 4), 4)
        right = self._ts_rank(self._decay_linear(self._ts_argmax(right_corr, 13), 14), 13)
        return -self._max2(left, right)
    
    def alpha097(self) -> pd.Series:
        d = self.data
        blend = d["low"] * 0.721001 + d["vwap"] * (1.0 - 0.721001)
        left = self._rank(self._decay_linear(self._delta(self._neutralize(blend, "industry"), 3), 20))
        corr = self._corr(self._ts_rank(d["low"], 8), self._ts_rank(self._adv(60), 17), 5)
        right = self._ts_rank(self._decay_linear(self._ts_rank(corr, 19), 16), 7)
        return -(left - right)

    def alpha098(self) -> pd.Series:
        d = self.data
        left = self._rank(self._decay_linear(self._corr(d["vwap"], self._sum(self._adv(5), 26), 5), 7))
        corr = self._corr(self._rank(d["open"]), self._rank(self._adv(15)), 21)
        right = self._rank(self._decay_linear(self._ts_rank(self._ts_argmin(corr, 9), 7), 8))
        return left - right
    
    def alpha099(self) -> pd.Series:
        d = self.data
        left = self._rank(self._corr(self._sum((d["high"] + d["low"]) / 2, 20), self._sum(self._adv(60), 20), 9))
        right = self._rank(self._corr(d["low"], d["volume"], 6))
        return -(left < right).astype(float).where(left.notna() & right.notna())
    
    
    def alpha100(self) -> pd.Series:
        d = self.data
        location = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["high"] - d["low"] + self.eps)
        p0 = 1.5 * self._scale(self._neutralize(self._neutralize(self._rank(location * d["volume"]), "subindustry"), "subindustry"))
        p1_raw = self._corr(d["close"], self._rank(self._adv(20)), 5) - self._rank(self._ts_argmin(d["close"], 30))
        p1 = self._scale(self._neutralize(p1_raw, "subindustry"))
        return -(p0 - p1) * d["volume"] / (self._adv(20) + self.eps)
    
    def alpha101(self) -> pd.Series:
        d = self.data
        return (d["close"] - d["open"]) / (d["high"] - d["low"] + 0.001)
    
    

        

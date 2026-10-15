import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(slots=True)
class BarData:
    """Bar representation used by offline factor data loaders."""

    symbol: str
    datetime: datetime
    frequency: str
    open: float
    high: float
    low: float
    close: float
    preclose: float
    volume: float
    amount: float
    adjustflag: int
    turn: float
    tradestatus: int
    pct_chg: float
    pe_ttm: float
    pb_mrq: float
    ps_ttm: float
    pcf_ncf_ttm: float
    is_st: int


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


class FactorDataBuilder:
    """Build generic bar inputs for factors supplied by a research workflow."""

    def __init__(self, bar_cache):
        self.bar_cache = bar_cache

    def build_symbol_data_map(self, symbols: list[str], frequency: str = "60s", count: int = 21) -> Dict[str, List]:
        result = {}
        for symbol in symbols:
            bars = self.bar_cache.get_bars(symbol=symbol, frequency=frequency, count=count)
            if len(bars) >= count:
                result[symbol] = bars
        return result

    def load_bars_from_csv(
        self,
        file_path: str | Path,
        frequency: str = "1d",
        only_trading: bool = True,
        exclude_st: bool = False,
    ) -> List[BarData]:
        bars: List[BarData] = []
        with Path(file_path).open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                trade_status = _to_int(row.get("tradestatus"))
                is_st = _to_int(row.get("isST"))
                if only_trading and trade_status != 1:
                    continue
                if exclude_st and is_st == 1:
                    continue
                bars.append(BarData(
                    symbol=row["code"], datetime=datetime.strptime(row["date"], "%Y-%m-%d"), frequency=frequency,
                    open=_to_float(row.get("open")), high=_to_float(row.get("high")), low=_to_float(row.get("low")),
                    close=_to_float(row.get("close")), preclose=_to_float(row.get("preclose")),
                    volume=_to_float(row.get("volume")), amount=_to_float(row.get("amount")),
                    adjustflag=_to_int(row.get("adjustflag")), turn=_to_float(row.get("turn")),
                    tradestatus=trade_status, pct_chg=_to_float(row.get("pctChg")), pe_ttm=_to_float(row.get("peTTM")),
                    pb_mrq=_to_float(row.get("pbMRQ")), ps_ttm=_to_float(row.get("psTTM")),
                    pcf_ncf_ttm=_to_float(row.get("pcfNcfTTM")), is_st=is_st,
                ))
        return sorted(bars, key=lambda item: item.datetime)

    def build_symbol_data_map_from_csv(
        self,
        file_path: str | Path,
        frequency: str = "1d",
        count: Optional[int] = None,
        only_trading: bool = True,
        exclude_st: bool = False,
    ) -> Dict[str, List[BarData]]:
        grouped: Dict[str, List[BarData]] = defaultdict(list)
        for bar in self.load_bars_from_csv(file_path, frequency, only_trading, exclude_st):
            grouped[bar.symbol].append(bar)
        result: Dict[str, List[BarData]] = {}
        for symbol, bars in grouped.items():
            bars.sort(key=lambda item: item.datetime)
            if count is not None:
                if len(bars) < count:
                    continue
                bars = bars[-count:]
            result[symbol] = bars
        return result

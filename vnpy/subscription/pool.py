from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Any, Dict, List, Optional


@dataclass
class PoolItem:
    """
    股票池中的单个股票定义。
    """

    symbol: str
    name: str
    sector: str = ""
    tier: str = "watch"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


class SubscriptionPool:
    def __init__(self, items: Optional[List[PoolItem]] = None) -> None:
        self._items: Dict[str, PoolItem] = {}
        if items:
            for item in items:
                self.add(item)

    def add(self, item: PoolItem) -> None:
        self._items[item.symbol] = item

    def remove(self, symbol: str) -> None:
        if symbol in self._items:
            del self._items[symbol]

    def get(self, symbol: str) -> Optional[PoolItem]:
        return self._items.get(symbol)

    def all_items(self) -> List[PoolItem]:
        """
        获取全部股票。
        """
        return list(self._items.values())

    def enabled_items(self) -> List[PoolItem]:
        """
        获取启用状态的股票。
        """
        return [item for item in self._items.values() if item.enabled]

    def symbols(self, only_enabled: bool = True) -> List[str]:
        """
        获取股票代码列表。
        """
        items = self.enabled_items() if only_enabled else self.all_items()
        return [item.symbol for item in items]

    def subscribe_symbols(self, only_enabled: bool = True) -> str:
        """
        转成掘金 subscribe 用的字符串
        """
        return ",".join(self.symbols(only_enabled=only_enabled))

    def by_sector(self, sector: str, only_enabled: bool = True) -> List[PoolItem]:
        """
        按行业获取股票
        """
        items = self.enabled_items() if only_enabled else self.all_items()
        return [item for item in items if item.sector == sector]

    def by_tier(self, tier: str, only_enabled: bool = True) -> List[PoolItem]:
        """
        按层级获取股票，比如 core/watch
        """
        items = self.enabled_items() if only_enabled else self.all_items()
        return [item for item in items if item.tier == tier]

    def sectors(self) -> List[str]:
        """
        获取全部行业名称
        """
        return sorted({item.sector for item in self._items.values() if item.sector})

    def enable(self, symbol: str) -> None:
        item = self.get(symbol)
        if item:
            item.enabled = True

    def disable(self, symbol: str) -> None:
        item = self.get(symbol)
        if item:
            item.enabled = False


def baostock_to_gm_symbol(code: str) -> str:
    """
    将 BaoStock 代码转换成掘金订阅代码。

    BaoStock: sh.600519 / sz.000001
    掘金:     SHSE.600519 / SZSE.000001
    """
    market, raw_symbol = code.split(".", 1)

    if market == "sh":
        return f"SHSE.{raw_symbol}"

    if market == "sz":
        return f"SZSE.{raw_symbol}"

    raise ValueError(f"unsupported BaoStock market code: {code}")


def _query_date_or_today(query_date: Optional[str] = None) -> str:
    if query_date:
        return query_date

    return datetime.today().strftime("%Y-%m-%d")


def _build_industry_map(client: Any) -> Dict[str, str]:
    if not hasattr(client, "query_stock_industry"):
        return {}

    industry_df = client.query_stock_industry()

    if industry_df is None or industry_df.empty or "code" not in industry_df.columns:
        return {}

    industry_map: Dict[str, str] = {}

    for _, row in industry_df.iterrows():
        code = row.get("code", "")
        industry = row.get("industry", "")
        if code:
            industry_map[str(code)] = "" if industry is None else str(industry)

    return industry_map


def create_baostock_pool(query_date: Optional[str] = None, only_trading: bool = True, include_industry: bool = True,) -> SubscriptionPool:
    """
    从 BaoStock 生成沪深 A 股股票池。
    """
    from vnpy.datafeed.BaostockFinancialClient import BaostockFinancialClient

    actual_query_date = _query_date_or_today(query_date)

    with BaostockFinancialClient() as client:
        stocks = client.get_a_stocks(query_date=actual_query_date, only_trading=only_trading)

        if include_industry:
            industry_map = _build_industry_map(client)
        else:
            industry_map = {}

    if stocks is None or stocks.empty:
        return SubscriptionPool([])

    items: List[PoolItem] = []

    skipped_empty_code = 0
    convert_failed = 0


    for index, row in stocks.iterrows():
        code = str(row.get("code", "")).strip()
        if not code:
            skipped_empty_code += 1
            continue

        try:
            symbol = baostock_to_gm_symbol(code)
        except Exception as e:
            convert_failed += 1
            continue

        name = str(row.get("code_name", ""))
        sector = industry_map.get(code, "")

        items.append(
            PoolItem(
                symbol=symbol,
                name=name,
                sector=sector,
                tier="watch",
                enabled=True,
                tags=["baostock"],
            )
        )
    return SubscriptionPool(items)


def create_default_pool() -> SubscriptionPool:
    return create_baostock_pool()

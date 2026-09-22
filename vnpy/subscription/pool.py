from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PoolItem:
    """
    股票池中的单个股票定义
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
        获取全部股票
        """
        return list(self._items.values())

    def enabled_items(self) -> List[PoolItem]:
        """
        获取启用状态的股票
        """
        return [item for item in self._items.values() if item.enabled]

    def symbols(self, only_enabled: bool = True) -> List[str]:
        """
        获取股票代码列表
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

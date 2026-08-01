from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from vnpy.datafeed.model import BarData, BarSource, normalize_bar


DEFAULT_FIELDS = "symbol,bob,open,high,low,close,volume,amount"


class GmLocalDataFeed:
    """Read downloaded GM history through GM's local data service."""

    def __init__(
        self,
        token: str | None = None,
        history_func: Callable[..., Any] | None = None,
        history_n_func: Callable[..., Any] | None = None,
        set_token_func: Callable[[str], Any] | None = None,
    ) -> None:
        if history_func is None or history_n_func is None or set_token_func is None:
            try:
                from gm.api import history, history_n, set_token
            except ImportError as exc:
                raise RuntimeError(
                    "gm.api is required to read the GM local data store"
                ) from exc

            history_func = history_func or history
            history_n_func = history_n_func or history_n
            set_token_func = set_token_func or set_token

        self._history = history_func
        self._history_n = history_n_func
        self._set_token = set_token_func

        if token:
            self._set_token(token)

    def load_history(
        self,
        symbols: str | Iterable[str],
        frequency: str,
        start: str,
        end: str,
        fields: str = DEFAULT_FIELDS,
        adjust: int = 0,
    ) -> list[BarData]:
        """Load bars in a time range from the GM local data service."""
        symbol_text = self._normalize_symbols(symbols)
        raw_bars = self._history(
            symbol=symbol_text,
            frequency=frequency,
            start_time=start,
            end_time=end,
            fields=fields,
            adjust=adjust,
            df=True,
        )
        return self._convert(raw_bars, frequency)

    def load_recent(
        self,
        symbols: str | Iterable[str],
        frequency: str,
        count: int,
        end: str | None = None,
        fields: str = DEFAULT_FIELDS,
        adjust: int = 0,
    ) -> list[BarData]:
        """Load the latest N bars from the GM local data service."""
        if count <= 0:
            raise ValueError("count must be greater than zero")

        symbol_text = self._normalize_symbols(symbols)
        raw_bars = self._history_n(
            symbol=symbol_text,
            frequency=frequency,
            count=count,
            end_time=end,
            fields=fields,
            adjust=adjust,
            df=True,
        )
        return self._convert(raw_bars, frequency)

    @staticmethod
    def _normalize_symbols(symbols: str | Iterable[str]) -> str:
        if isinstance(symbols, str):
            values = symbols.split(",")
        else:
            values = symbols

        normalized = [str(symbol).strip() for symbol in values if str(symbol).strip()]
        if not normalized:
            raise ValueError("at least one GM symbol is required")
        return ",".join(normalized)

    @staticmethod
    def _convert(raw_bars: Any, frequency: str) -> list[BarData]:
        if raw_bars is None:
            return []

        if hasattr(raw_bars, "to_dict"):
            records = raw_bars.to_dict("records")
        elif isinstance(raw_bars, Mapping):
            records = [raw_bars]
        else:
            records = list(raw_bars)

        bars = [
            normalize_bar(record, frequency=frequency, source=BarSource.GM_LOCAL)
            for record in records
        ]
        bars.sort(key=lambda bar: (bar.bob, bar.symbol))
        return bars

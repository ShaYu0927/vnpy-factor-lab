from pathlib import Path
from typing import Any

import pandas as pd

from vnpy.datafeed.BarCache import BarData


class CsvBarDataFeed:
    """Load local CSV bar data into BarData objects."""

    symbol_columns = ("symbol", "vt_symbol", "code")
    datetime_columns = ("bob", "datetime", "time", "date")
    open_columns = ("open", "open_price")
    high_columns = ("high", "high_price")
    low_columns = ("low", "low_price")
    close_columns = ("close", "close_price")
    volume_columns = ("volume", "vol")
    amount_columns = ("amount", "turnover")

    def __init__(
        self,
        csv_path: str | Path,
        frequency: str = "60s",
        default_symbol: str | None = None,
        encoding: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.frequency = frequency
        self.default_symbol = default_symbol
        self.encoding = encoding

    def load_bars(self) -> list[BarData]:
        df = pd.read_csv(self.csv_path, encoding=self.encoding)
        return [self.row_to_bar(row) for _, row in df.iterrows()]

    def row_to_bar(self, row: pd.Series) -> BarData:
        symbol = self._get_value(row, self.symbol_columns, self.default_symbol)
        if not symbol:
            raise ValueError("CSV row is missing symbol/vt_symbol/code")

        dt_value = self._get_value(row, self.datetime_columns)
        if dt_value is None:
            raise ValueError("CSV row is missing bob/datetime/time/date")

        return BarData(
            symbol=str(symbol),
            bob=pd.to_datetime(dt_value).to_pydatetime(),
            open=float(self._get_value(row, self.open_columns)),
            high=float(self._get_value(row, self.high_columns)),
            low=float(self._get_value(row, self.low_columns)),
            close=float(self._get_value(row, self.close_columns)),
            volume=float(self._get_value(row, self.volume_columns, 0)),
            amount=float(self._get_value(row, self.amount_columns, 0)),
            frequency=self.frequency,
        )

    @staticmethod
    def _get_value(row: pd.Series, names: tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            if name in row and not pd.isna(row[name]):
                return row[name]
        return default

import csv
from datetime import datetime, timedelta
from io import StringIO
import sys

import polars as pl

from vnpy.config.runtime_config import ParquetConfig
from vnpy.datafeed.data_market_module import (
    load_market_snapshot, print_market_table, unregister_market_module,
)
from vnpy.datafeed.data_parquet_feed import ParquetDataFeed
from vnpy.event.engine import ModuleEngine


def test_print_market_table_outputs_every_row_and_column(tmp_path, monkeypatch):
    market = pl.DataFrame({
        'datetime': [datetime(2026, 1, 1) + timedelta(days=i) for i in range(12)],
        'vt_symbol': ['SHSE.600000'] * 12,
        'close': [10.0 + i for i in range(12)],
        'note': ['含\t制表符\n和换行'] + ['普通值'] * 11,
    })
    monkeypatch.setattr(ParquetDataFeed, '_resolve_day_bar_dir', staticmethod(lambda root: root))
    monkeypatch.setattr(ParquetDataFeed, 'load_frame', lambda self, **kwargs: market)
    engine = ModuleEngine()
    config = ParquetConfig(root=str(tmp_path), start='2026-01-01', end='2026-01-12')
    try:
        load_market_snapshot(engine, config)
        output = tmp_path / 'debug' / 'market.tsv'
        assert print_market_table(engine, output_path=output) == 12
        with output.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream, dialect='excel-tab'))
        assert rows[0] == market.columns
        assert len(rows) == 13
        assert rows[1][3] == '含\t制表符\n和换行'
        assert rows[-1][2] == '21.0'

        terminal = StringIO()
        monkeypatch.setattr(sys, 'stdout', terminal)
        assert print_market_table(engine) == 12
        assert list(csv.reader(StringIO(terminal.getvalue()), dialect='excel-tab')) == rows
    finally:
        unregister_market_module(engine)

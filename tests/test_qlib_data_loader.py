from datetime import datetime, timedelta

import numpy as np
import polars as pl

from vnpy.alpha.dataset.datasets import Basic3DL
from vnpy.alpha.dataset.loader import translate_qlib_expression


def make_frame() -> pl.DataFrame:
    start = datetime(2025, 1, 1)
    rows = []
    for offset in range(22):
        for symbol_index, symbol in enumerate(("AAA.SSE", "BBB.SSE")):
            rows.append({
                "datetime": start + timedelta(days=offset),
                "vt_symbol": symbol,
                "close": 100.0 + symbol_index * 10 + offset,
                "volume": 1_000.0 + symbol_index * 100 + offset * 10,
            })
    return pl.DataFrame(rows)


def test_basic_three_uses_qlib_fields_and_names() -> None:
    fields, names = Basic3DL.get_feature_config()

    assert names == ["momentum_20", "volatility_20", "volume_20"]
    assert fields == [
        "$close/Ref($close, 20)-1",
        "Std($close/Ref($close, 1)-1, 20)",
        "$volume/(Mean($volume, 20)+1e-12)",
    ]


def test_translate_qlib_expression_to_local_operator_names() -> None:
    assert translate_qlib_expression(
        "Std($close/Ref($close, 1)-1, 20)"
    ) == "ts_std(close/ts_delay(close, 1)-1, 20)"


def test_basic_three_loader_builds_wide_feature_matrix() -> None:
    frame = make_frame()
    matrix = Basic3DL().load(frame)
    latest = matrix.filter(
        (pl.col("vt_symbol") == "AAA.SSE")
        & (pl.col("datetime") == datetime(2025, 1, 21))
    ).row(0, named=True)

    closes = np.arange(100.0, 121.0)
    returns = closes[1:] / closes[:-1] - 1
    volumes = np.arange(1_000.0, 1_210.0, 10.0)

    assert matrix.columns == [
        "datetime",
        "vt_symbol",
        "momentum_20",
        "volatility_20",
        "volume_20",
    ]
    assert np.isclose(latest["momentum_20"], closes[-1] / closes[0] - 1)
    assert np.isclose(latest["volatility_20"], np.std(returns, ddof=0))
    assert np.isclose(latest["volume_20"], volumes[-1] / volumes[-20:].mean())


def test_loader_preserves_official_instrument_key() -> None:
    matrix = Basic3DL(feature_names=["momentum_20"]).load(
        make_frame().rename({"vt_symbol": "instrument"})
    )

    assert matrix.columns == ["datetime", "instrument", "momentum_20"]


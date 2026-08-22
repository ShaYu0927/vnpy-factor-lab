from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from vnpy.datafeed.model import BarData


def write_kline_data(
    histories: Mapping[str, Sequence[BarData]],
    output: str | Path,
) -> Path:
    """Write browser-ready KLineCharts OHLCV data grouped by symbol."""
    payload = {
        symbol: [
            {
                "timestamp": int(bar.bob.timestamp() * 1000),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "turnover": bar.amount,
            }
            for bar in bars
        ]
        for symbol, bars in histories.items()
        if bars
    }
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_kline_report(
    histories: Mapping[str, Sequence[BarData]],
    output: str | Path,
    *,
    auto_open: bool = False,
) -> Path:
    """Write one interactive candlestick/volume report with a symbol selector."""
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    populated = [(symbol, list(bars)) for symbol, bars in histories.items() if bars]
    if not populated:
        raise ValueError("cannot create a K-line report without bars")

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
    for index, (symbol, bars) in enumerate(populated):
        visible = index == 0
        dates = [bar.bob for bar in bars]
        colors = [
            "#ef5350" if bar.close >= bar.open else "#26a69a"
            for bar in bars
        ]
        figure.add_trace(
            go.Candlestick(
                x=dates,
                open=[bar.open for bar in bars],
                high=[bar.high for bar in bars],
                low=[bar.low for bar in bars],
                close=[bar.close for bar in bars],
                name=symbol,
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
                visible=visible,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=dates,
                y=[bar.volume for bar in bars],
                marker_color=colors,
                name=f"{symbol} 成交量",
                visible=visible,
            ),
            row=2,
            col=1,
        )

    trace_count = len(populated) * 2
    buttons = []
    for index, (symbol, _bars) in enumerate(populated):
        visibility = [False] * trace_count
        visibility[index * 2:index * 2 + 2] = [True, True]
        buttons.append({
            "label": symbol,
            "method": "update",
            "args": [
                {"visible": visibility},
                {"title": {"text": f"{symbol} 日线行情"}},
            ],
        })

    first_symbol = populated[0][0]
    figure.update_layout(
        title={"text": f"{first_symbol} 日线行情", "x": 0.5},
        template="plotly_white",
        height=820,
        hovermode="x unified",
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 100, "b": 50},
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.12,
            "yanchor": "top",
        }],
    )
    figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    figure.update_yaxes(title_text="价格", row=1, col=1)
    figure.update_yaxes(title_text="成交量", row=2, col=1)

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, auto_open=auto_open)
    return path

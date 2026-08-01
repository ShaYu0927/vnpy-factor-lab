from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from vnpy.factor.core.factor_engine import FactorContext
from vnpy.factor.reversal import IntradayFadeEngineFactor


@dataclass(slots=True)
class CsvBar:
    date: str
    time: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


def load_bars(path: Path) -> list[CsvBar]:
    bars: list[CsvBar] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                CsvBar(
                    date=row["date"],
                    time=row["time"],
                    symbol=row["code"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    amount=float(row.get("amount") or 0.0),
                )
            )

    return bars


def analyze_bars(
    bars: list[CsvBar],
    volume_window: int,
) -> Iterable[dict[str, str | float | bool]]:
    if not bars:
        return

    factor = IntradayFadeEngineFactor(volume_window=volume_window)
    context = FactorContext()
    symbol = bars[0].symbol

    for index in range(factor.min_bars - 1, len(bars)):
        history = bars[: index + 1]
        result = factor.calculate(symbol, history, context)

        if result is None:
            continue

        bar = bars[index]
        yield {
            "date": bar.date,
            "time": bar.time,
            "symbol": result.symbol,
            "close": bar.close,
            "factor": result.factor,
            "rise": result.rise,
            "fall_back": result.fall_back,
            "fall_ratio": result.fall_ratio,
            "volume_ratio": result.volume_ratio,
            "signal": result.signal,
        }


def write_rows(path: Path, rows: Iterable[dict[str, str | float | bool]]) -> int:
    fieldnames = [
        "date",
        "time",
        "symbol",
        "close",
        "factor",
        "rise",
        "fall_back",
        "fall_ratio",
        "volume_ratio",
        "signal",
    ]

    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze intraday fade factor from a CSV file.")
    parser.add_argument("input", type=Path, help="Input CSV path, for example sh_600000_min5.csv")
    parser.add_argument("-o", "--output", type=Path, default=Path("intraday_fade_result.csv"))
    parser.add_argument("--volume-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bars = load_bars(args.input)
    rows = analyze_bars(bars, volume_window=args.volume_window)
    count = write_rows(args.output, rows)

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()

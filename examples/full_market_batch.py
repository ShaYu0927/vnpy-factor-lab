"""Run full-market history calculation directly from the IDE or command line."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT))

from vnpy.main import main


if __name__ == "__main__":
    main(PROJECT / "config" / "runtime.full_market.json")

"""One chronological split over the observed session calendar of a stock panel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import math


@dataclass(frozen=True, slots=True)
class TimeSplitConfig:
    train_fraction: float = 0.7

    def __post_init__(self) -> None:
        value = self.train_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError("time_split.train_fraction must be between zero and one")


def parse_time_split_config(raw: dict) -> TimeSplitConfig:
    """Accept the old report setting, but reject conflicting duplicate settings."""
    section = raw.get("time_split", {})
    evaluation = raw.get("factor_evaluation", {})
    if not isinstance(section, dict) or not isinstance(evaluation, dict):
        raise ValueError("time_split and factor_evaluation must be objects")
    if set(section) - {"train_fraction"}:
        raise ValueError("unknown time_split option")
    legacy = evaluation.get("train_fraction")
    fraction = section.get("train_fraction", legacy if legacy is not None else 0.7)
    config = TimeSplitConfig(fraction)
    if legacy is not None:
        TimeSplitConfig(legacy)
        if legacy != fraction:
            raise ValueError(
                "conflicting train_fraction settings; keep only time_split.train_fraction"
            )
    return config


@dataclass(frozen=True, slots=True)
class SessionSplit:
    sessions: tuple[date | datetime, ...]
    train_count: int

    def __post_init__(self) -> None:
        if len(self.sessions) < 2 or not 1 <= self.train_count < len(self.sessions):
            raise ValueError(
                "time split requires at least one training and one holdout session"
            )
        if tuple(sorted(set(self.sessions))) != self.sessions:
            raise ValueError("session calendar must be sorted and unique")

    @property
    def holdout_start(self) -> date | datetime:
        return self.sessions[self.train_count]

    def dates_for(self, segment: str) -> tuple[date | datetime, ...]:
        if segment == "train":
            return self.sessions[: self.train_count]
        if segment == "holdout":
            return self.sessions[self.train_count :]
        if segment == "all":
            return self.sessions
        raise ValueError(f"unknown segment: {segment}")

    def as_dict(self) -> dict:
        dates = [day.isoformat() for day in self.sessions]
        return {
            "calendar": "sorted unique observed dates across the configured stock panel",
            "calendar_sha256": sha256("\n".join(dates).encode()).hexdigest(),
            "sessions": dates,
            "train": {
                "start": dates[0],
                "end": dates[self.train_count - 1],
                "sessions": self.train_count,
            },
            "holdout": {
                "start": dates[self.train_count],
                "end": dates[-1],
                "sessions": len(dates) - self.train_count,
            },
            "train_boundary": "signal_date < holdout_start and label_end_date < holdout_start",
        }


def build_session_split(
    dates: Iterable[date | datetime], config: TimeSplitConfig
) -> SessionSplit:
    sessions = tuple(sorted(set(dates)))
    return SessionSplit(sessions, int(len(sessions) * config.train_fraction))

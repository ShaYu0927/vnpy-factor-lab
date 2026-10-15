"""Configuration for bounded genetic factor search."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 100
    generations: int = 11
    elite_size: int = 10
    tournament_size: int = 3
    attempts_per_generation: int = 20_000
    period: int = 1          # 入场后持有的共享交易日数
    entry_offset: int = 1    # 信号收盘后延迟的共享交易日数
    train_fraction: float = 0.7
    min_assets: int = 10
    min_days: int = 10
    holdout_top: int = 20
    seed: int = 42
    stability_weight: float = 0.25
    complexity_penalty: float = 0.01

    def __post_init__(self) -> None:
        integer_fields = (
            "population_size", "generations", "elite_size", "tournament_size",
            "attempts_per_generation", "period", "entry_offset", "min_assets",
            "min_days", "holdout_top",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.elite_size >= self.population_size:
            raise ValueError("elite_size must be smaller than population_size")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size must not exceed population_size")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.train_fraction, (int, float)) or not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        for name in ("stability_weight", "complexity_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative finite number")

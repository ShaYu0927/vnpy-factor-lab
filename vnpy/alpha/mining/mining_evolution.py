"""Run genetic factor search against the configured Parquet market sample."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from vnpy.alpha.mining.mining_config import runtime_iteration_config
from vnpy.alpha.mining.ga import (
    EvolutionConfig,
    EvolutionEngine,
    Fitness,
    FitnessEvaluator,
    Individual,
)
from vnpy.alpha.alpha_domain import MarketFrame, SessionSplit
from vnpy.config.config_runtime_config import load_runtime_config
from vnpy.datafeed.datafeed_data_parquet_feed import ParquetDataFeed


@dataclass(frozen=True)
class EvolutionResult:
    """Results available to the caller without creating run files."""

    split: SessionSplit
    candidates: tuple[Individual, ...]
    seed_admission: tuple[dict, ...]
    holdout: tuple[tuple[Individual, Fitness], ...]


def run(
    config_path: str | Path, *, population: int = 100, generations: int = 11
) -> EvolutionResult:
    """Search on training data and evaluate its leaders on holdout in memory."""
    source = Path(config_path).expanduser().resolve()
    runtime = load_runtime_config(source)
    iteration = runtime_iteration_config(
        runtime.raw.get("expression_iteration", {}), source.parent
    )
    if iteration is None:
        raise ValueError("expression_iteration must be enabled for genetic search")

    options = EvolutionConfig(
        population_size=population,
        generations=generations,
        train_fraction=runtime.time_split.train_fraction,
    )
    setting = runtime.parquet
    market = ParquetDataFeed(setting.root).load_frame(
        start=setting.start,
        end=setting.end,
        symbols=setting.symbols,
        markets=setting.markets,
        frequency=setting.frequency,
        skip_zero_volume=setting.skip_zero_volume,
        skip_invalid_ohlc=setting.skip_invalid_ohlc,
        allow_missing_years=setting.allow_missing_years,
    )
    if market.is_empty():
        raise ValueError("no market data available for genetic search")
    market = MarketFrame(market)
    split = market.split(runtime.time_split)
    evaluator = FitnessEvaluator(market.frame, options, split=split)
    seed_alphas = runtime.raw.get("alphas", [])
    if not isinstance(seed_alphas, list):
        raise ValueError("alphas must be a list of name/formula objects")
    engine = EvolutionEngine(
        iteration.search_space, options, evaluator, seed_alphas=seed_alphas
    )

    candidates = tuple(engine.run())
    leaders = sorted(candidates, key=lambda item: item.fitness.score, reverse=True)[
        : options.holdout_top
    ]
    holdout = tuple(
        (item, evaluator.score(item.candidate.tree, holdout=True)) for item in leaders
    )
    return EvolutionResult(split, candidates, tuple(engine.seed_admission), holdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_runtime_basic_alphas.json")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=11)
    args = parser.parse_args()
    result = run(args.config, population=args.population, generations=args.generations)
    admitted = sum(row["status"] == "admitted" for row in result.seed_admission)
    print(
        f"train={result.split.train_count} sessions, "
        f"holdout={len(result.split.dates_for('holdout'))} sessions, "
        f"seeds={admitted}/{len(result.seed_admission)}, "
        f"candidates={len(result.candidates)}, "
        f"best_training_score={max(item.fitness.score for item in result.candidates):.6f}"
    )


if __name__ == "__main__":
    main()

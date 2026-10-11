"""Exercise genetic search boundaries and its real training scorer."""

from datetime import datetime, timedelta
from random import Random

import polars as pl
import pytest

from vnpy.alpha.expression import ConstantNode, ExpressionAnalyzer, FieldNode, OperatorNode, WindowNode
from vnpy.alpha.mining.config import SearchSpace
from vnpy.alpha.mining.ga import (
    Crossover, EvolutionConfig, EvolutionEngine, Fitness, FitnessEvaluator,
    Individual, Mutator, Population, Selector,
)
from vnpy.alpha.mining.ga.fitness import _labels
from vnpy.alpha.mining.schema import Candidate


def market_frame() -> pl.DataFrame:
    first = datetime(2026, 1, 1)
    rows = []
    for day in range(28):
        for asset in range(12):
            # Omit one entry day to check that the label uses the shared
            # session calendar rather than the next observed bar for an asset.
            if day == 5 and asset == 0:
                continue
            close = 10 + asset * 0.3 + day * (0.02 + asset * 0.003)
            rows.append({
                "datetime": first + timedelta(days=day), "vt_symbol": f"S{asset}",
                "open": close * 0.99, "high": close * 1.01,
                "low": close * 0.98, "close": close, "volume": 1000 + asset,
            })
    return pl.DataFrame(rows)


def test_labels_follow_shared_sessions_and_training_boundary() -> None:
    frame = market_frame()
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=4)
    labels, split = _labels(frame, options)
    missing_entry = labels.filter(
        (pl.col("vt_symbol") == "S0")
        & (pl.col("datetime") == datetime(2026, 1, 5))
    )
    assert missing_entry.is_empty()
    scorer = FitnessEvaluator(frame, options)
    assert scorer.split_date == split
    assert scorer.training_labels.filter(pl.col("exit_date") >= split).is_empty()
    assert scorer.holdout_labels.filter(pl.col("datetime") < split).is_empty()


def test_holdout_price_change_does_not_change_training_fitness() -> None:
    frame = market_frame()
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=4)
    baseline = FitnessEvaluator(frame, options)
    changed = frame.with_columns(
        pl.when(pl.col("datetime") >= baseline.split_date)
        .then(pl.col("close") * (1 + pl.col("vt_symbol").str.slice(1).cast(pl.Int64) * 0.03))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    tree = OperatorNode("ts_delta", (FieldNode("close"), WindowNode(1)))
    assert FitnessEvaluator(changed, options).score(tree) == baseline.score(tree)


def test_population_selector_and_tree_edits_have_separate_contracts() -> None:
    trees = (FieldNode("close"), FieldNode("open"), FieldNode("volume"))
    members = tuple(
        Individual(Candidate(tree, 1, 1, (tree.name,)), Fitness(score, None, 0, 0), "random")
        for tree, score in zip(trees, (0.1, 0.3, 0.2))
    )
    population = Population(1, members)
    assert population.elites(1) == (members[1],)
    assert Selector(Random(5), 3).select(population) == members[1]
    with pytest.raises(ValueError, match="duplicate"):
        Population(1, (members[0], members[0]))

    space = SearchSpace(fields=("close", "volume"), windows=(1, 3))
    assert Mutator(space, Random(5)).mutate(FieldNode("close")) == FieldNode("volume")
    left = OperatorNode("add", (FieldNode("close"), FieldNode("close")))
    right = OperatorNode("mul", (FieldNode("volume"), FieldNode("volume")))
    crossed = Crossover(Random(5)).cross(left, right)
    assert crossed is not None
    assert {node.name for node in crossed.walk() if isinstance(node, FieldNode)} == {"close", "volume"}


def test_fitness_exposes_stability_and_penalizes_equivalent_complex_tree() -> None:
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=4)
    evaluator = FitnessEvaluator(market_frame(), options)
    simple = evaluator.score(FieldNode("close"))
    complex_same_values = evaluator.score(OperatorNode("add", (FieldNode("close"), ConstantNode(0))))
    assert simple.ic_days >= options.min_days
    assert simple.rank_ic == pytest.approx(complex_same_values.rank_ic)
    assert simple.stability is not None
    assert simple.complexity == 1
    assert complex_same_values.complexity == 3
    assert simple.score > complex_same_values.score


def test_genetic_search_produces_unique_scored_offspring() -> None:
    space = SearchSpace(fields=("close", "open", "volume"), windows=(1, 3))
    options = EvolutionConfig(population_size=8, generations=3, elite_size=2,
                              tournament_size=3, min_assets=3, min_days=4)
    search = EvolutionEngine(space, options, FitnessEvaluator(market_frame(), options))
    history = search.run()
    assert len(history) == 8 + 2 * 6
    assert len({item.candidate.fingerprint for item in history}) == len(history)
    assert any(item.origin in {"mutation", "crossover"} for item in history[8:])
    assert search.population.generation == 3
    assert len(search.population.members) == options.population_size
    for item in history:
        assert item.fitness.score >= 0
        assert item.candidate.tree.depth <= space.max_depth
        assert ExpressionAnalyzer().analyze(item.candidate.tree).lookback <= space.max_lookback


def test_invalid_evolution_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="elite_size"):
        EvolutionConfig(population_size=10, elite_size=10)

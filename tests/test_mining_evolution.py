"""Exercise genetic search boundaries and its real training scorer."""

from datetime import datetime, timedelta
from random import Random

import polars as pl
import pytest

from vnpy.alpha.expression import ConstantNode, ExpressionAnalyzer, FieldNode, OperatorNode, WindowNode
from vnpy.alpha.mining.mining_config import SearchSpace
from vnpy.alpha.mining.ga import (
    Crossover, EvolutionConfig, EvolutionEngine, Fitness, FitnessEvaluator,
    Individual, Mutator, Population, Selector,
)
from vnpy.alpha.mining.ga.ga_fitness import _labels
from vnpy.alpha.mining.mining_schema import Candidate


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


def ranked_market(rates: tuple[float, ...], *, reverse_holdout: bool = False) -> pl.DataFrame:
    """Keep factor ranks fixed while controlling next-session return ranks."""
    first = datetime(2026, 1, 1)
    closes = [100.0] * len(rates)
    rows = []
    for day in range(14):
        if day:
            current_rates = rates[::-1] if reverse_holdout and day > 10 else rates
            closes = [close * (1 + rate) for close, rate in zip(closes, current_rates, strict=True)]
        for asset, close in enumerate(closes):
            rows.append({
                "datetime": first + timedelta(days=day), "vt_symbol": f"S{asset}",
                "open": close, "high": close, "low": close, "close": close,
                "volume": 100 + asset,
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
    first_label = labels.filter(
        (pl.col("datetime") == datetime(2026, 1, 1))
        & (pl.col("vt_symbol") == "S0")
    ).row(0, named=True)
    assert first_label["entry_date"] == datetime(2026, 1, 2)
    assert first_label["exit_date"] == datetime(2026, 1, 3)


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
        for tree, score in zip(trees, (0.1, 0.3, 0.2), strict=True)
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


def test_rank_ic_sign_and_holdout_direction_are_fixed_by_training() -> None:
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=2)
    tree = FieldNode("volume")
    positive = FitnessEvaluator(ranked_market((0.01, 0.02, 0.03), reverse_holdout=True), options)
    training = positive.score(tree)
    holdout = positive.score(tree, holdout=True)
    negative = FitnessEvaluator(ranked_market((0.03, 0.02, 0.01)), options).score(tree)

    assert training.rank_ic == pytest.approx(1)
    assert training.direction == 1
    assert training.coverage == 1
    assert training.outcome == "scored"
    assert negative.rank_ic == pytest.approx(-1)
    assert negative.direction == -1
    assert negative.score == pytest.approx(training.score)
    assert holdout.rank_ic == pytest.approx(-1)
    assert holdout.direction == 1
    assert holdout.score < 0


def test_rank_ic_uses_average_ranks_for_ties() -> None:
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=2)
    frame = ranked_market((0.01, 0.02, 0.02)).with_columns(
        pl.when(pl.col("vt_symbol") == "S2").then(101).otherwise(pl.col("volume")).alias("volume")
    )
    fitness = FitnessEvaluator(frame, options).score(FieldNode("volume"))
    assert fitness.rank_ic == pytest.approx(1)
    assert fitness.outcome == "scored"


def test_fitness_distinguishes_zero_ic_missing_samples_and_calculation_error() -> None:
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=4, min_days=2)
    evaluator = FitnessEvaluator(ranked_market((0.02, 0.04, 0.01, 0.03)), options)
    low = evaluator.score(FieldNode("volume"))
    insufficient = evaluator.score(OperatorNode("sub", (FieldNode("close"), FieldNode("close"))))
    failed = evaluator.score(FieldNode("unknown_field"))

    assert low.outcome == "scored"
    assert low.rank_ic == pytest.approx(0)
    assert low.score == pytest.approx(0)
    assert low.error is None
    assert insufficient.outcome == "insufficient_samples"
    assert insufficient.ic_days == 0
    assert insufficient.error is None
    assert failed.outcome == "calculation_error"
    assert failed.error


def test_labels_reject_duplicate_market_keys() -> None:
    frame = market_frame()
    options = EvolutionConfig(population_size=8, elite_size=2, min_assets=3, min_days=4)
    with pytest.raises(ValueError, match="duplicate"):
        FitnessEvaluator(pl.concat((frame, frame.head(1))), options)


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


def test_configured_alphas_seed_first_generation_and_random_candidates_fill_it() -> None:
    space = SearchSpace(fields=("close", "open", "volume"), windows=(1, 3))
    options = EvolutionConfig(population_size=8, generations=2, elite_size=2,
                              min_assets=3, min_days=4)
    seeds = [
        {"name": "price", "formula": "close"},
        {"name": "volume", "formula": "volume"},
        {"name": "duplicate", "formula": "close"},
        {"name": "unknown", "formula": "missing_field"},
        {"name": "constant", "formula": "close - close"},
        {"name": "bad_formula", "formula": "close /"},
    ]
    search = EvolutionEngine(space, options, FitnessEvaluator(market_frame(), options),
                             seed_alphas=seeds)
    history = search.run()
    first = history[:options.population_size]
    seeded = [item for item in first if item.origin == "seed"]
    assert {item.seed_name for item in seeded} == {"price", "volume"}
    assert all(item.fitness.outcome == "scored" for item in seeded)
    assert len(first) == options.population_size
    assert sum(item.origin == "random" for item in first) == options.population_size - len(seeded)
    assert all(item.fitness.outcome == "scored" for item in first)
    assert len({item.candidate.fingerprint for item in history}) == len(history)
    assert seeded[0].record()["seed_name"] in {"price", "volume"}
    admissions = {item["name"]: item for item in search.seed_admission}
    assert admissions["price"]["status"] == "admitted"
    assert admissions["volume"]["status"] == "admitted"
    assert "duplicate" in admissions["duplicate"]["reason"]
    assert "unknown field" in admissions["unknown"]["reason"]
    assert "insufficient_samples" in admissions["constant"]["reason"]
    assert "invalid formula" in admissions["bad_formula"]["reason"]


def test_seed_admission_respects_population_capacity() -> None:
    space = SearchSpace(fields=("close", "open", "volume"))
    options = EvolutionConfig(population_size=2, generations=1, elite_size=1,
                              tournament_size=2, min_assets=3, min_days=4)
    seeds = [{"name": name, "formula": name} for name in ("close", "open", "volume")]
    search = EvolutionEngine(space, options, FitnessEvaluator(market_frame(), options),
                             seed_alphas=seeds)
    history = search.run()
    assert len(history) == 2
    assert all(item.origin == "seed" for item in history)
    assert [item["status"] for item in search.seed_admission].count("admitted") == 2
    assert [item["status"] for item in search.seed_admission].count("rejected") == 1


def test_invalid_evolution_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="elite_size"):
        EvolutionConfig(population_size=10, elite_size=10)

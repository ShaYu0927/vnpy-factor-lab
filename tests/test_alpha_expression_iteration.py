from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import random

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine
from vnpy.alpha.expression import ExpressionAnalyzer, ExpressionParser, PolarsCompiler, PolarsExecutor
from vnpy.alpha.mining import (
    ExpressionGenerator, ExpressionIterationConfig, SearchSpace, iterate_expressions,
    load_iteration_config, run_expression_iteration, tree_fingerprint,
)


@pytest.mark.parametrize("kwargs", [
    {"fields": []}, {"fields": ["close", "label"]}, {"fields": ["future_return"]},
    {"fields": ["close", "close"]}, {"operators": ["Rank"]}, {"operators": ["unknown"]},
    {"operators": ["Mean", "ts_mean"]}, {"operators": "Mean"},
    {"windows": [0]}, {"windows": [True]}, {"windows": [1.5]}, {"windows": []},
    {"constants": [float("nan")]}, {"constants": [True]},
    {"max_depth": 0}, {"max_depth": 13}, {"max_nodes": 256},
    {"max_nodes": True}, {"max_lookback": -1},
])
def test_invalid_search_spaces_fail_early(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SearchSpace(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": 1.5}, {"batch_size": 0}, {"batches": -1},
    {"max_attempts_per_batch": 0}, {"output_root": ""}, {"search_space": {}},
])
def test_invalid_run_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        ExpressionIterationConfig(**kwargs)


def records(config):
    return [candidate.as_dict() for batch in iterate_expressions(config) for candidate in batch.candidates]


def test_seed_reproduces_all_batches_without_touching_global_random():
    config = ExpressionIterationConfig(batches=3, batch_size=15)
    before = random.getstate()
    first = records(config)
    assert first == records(config)
    assert first != records(replace(config, seed=43))
    assert random.getstate() == before
    assert len(first) == len({item["fingerprint"] for item in first}) == 45
    assert [item["batch"] for item in first] == [1] * 15 + [2] * 15 + [3] * 15


def test_generated_trees_obey_signatures_constraints_and_roundtrip():
    space = SearchSpace(
        operators=("add", "sub", "mul", "div", "Pow", "Abs", "Sign", "Log",
                   "Ref", "Delta", "Mean", "Sum", "Std", "Min", "Max", "Corr"),
        windows=(1, 2, 5), max_depth=5, max_nodes=16, max_lookback=8,
    )
    generator = ExpressionGenerator(space, seed=123)
    parser = ExpressionParser()
    for _ in range(4):
        batch = generator.generate_batch(40, max_attempts=2000)
        assert batch.complete
        assert batch.attempts == len(batch.candidates) + batch.duplicates + sum(batch.rejected.values())
        for candidate in batch.candidates:
            tree = candidate.tree
            assert tree.depth <= space.max_depth
            assert sum(1 for _ in tree.walk()) <= space.max_nodes
            assert 1 <= candidate.lookback <= space.max_lookback
            assert tree == parser.parse(candidate.formula)
            assert tree_fingerprint(tree) == tree_fingerprint(parser.parse(candidate.formula))
            analysis = ExpressionAnalyzer().analyze(tree, set(space.fields))
            assert analysis.fields and not analysis.uses_cross_section
            assert analysis.lookback == candidate.lookback


def test_budget_exhaustion_returns_partial_results_and_stops_batches():
    config = ExpressionIterationConfig(
        search_space=SearchSpace(fields=("close",), operators=(), max_depth=1),
        batches=5, batch_size=3, max_attempts_per_batch=7,
    )
    batches = list(iterate_expressions(config))
    assert len(batches) == 1
    batch = batches[0]
    assert not batch.complete
    assert [item.formula for item in batch.candidates] == ["close"]
    assert batch.attempts == 7 and batch.duplicates == 6


def test_rejected_tree_is_not_recompiled_in_later_attempts(monkeypatch):
    generator = ExpressionGenerator(SearchSpace(max_lookback=2))
    tree = ExpressionParser().parse("Ref(close, 20)")
    monkeypatch.setattr(generator, "_grow", lambda *args, **kwargs: tree)
    batch = generator.generate_batch(2, max_attempts=4)
    assert not batch.candidates
    assert batch.rejected == {"lookback_limit": 1}
    assert batch.duplicates == 3


def market_frame():
    return pl.DataFrame([
        {
            "datetime": datetime(2026, 1, 1) + timedelta(days=i), "vt_symbol": symbol,
            "open": base + i * 0.4, "high": base + i * 0.4 + 2,
            "low": base + i * 0.4 - 2, "close": base + i * 0.4 + (i % 3) * 0.2,
            "volume": float(100 + i * 3 + i % 7),
        }
        for symbol, base in (("A", 10), ("B", 30)) for i in reversed(range(65))
    ])


def test_recorded_formulas_reload_and_calculate_with_existing_engine(tmp_path):
    config = ExpressionIterationConfig(batches=2, batch_size=12, output_root=str(tmp_path))
    result = run_expression_iteration(config)
    payload = json.loads((result.run_path / "alphas.generated.json").read_text(encoding="utf-8"))
    alphas = [Alpha(**item) for item in payload["alphas"]]
    frame = market_frame()
    calculated = AlphaEngine(alphas).calculate(frame)
    assert calculated.height == frame.height
    assert calculated.width == len(alphas) + 2
    for alpha in alphas:
        compiled = PolarsCompiler().compile(alpha.parse())
        expected = PolarsExecutor(frame).run(compiled, alpha.name)
        assert_frame_equal(calculated.select("datetime", "vt_symbol", alpha.name), expected)
    assert any(calculated[alpha.name].drop_nulls().len() for alpha in alphas)


def test_run_artifacts_are_reproducible_and_previous_run_is_preserved(tmp_path):
    config = ExpressionIterationConfig(batches=2, batch_size=5, output_root=str(tmp_path))
    first = run_expression_iteration(config)
    second = run_expression_iteration(config)
    assert first.run_path != second.run_path
    assert first.status == second.status == "complete"
    for name in ("candidates.jsonl", "batches.jsonl", "alphas.generated.json", "config.json"):
        assert (first.run_path / name).read_bytes() == (second.run_path / name).read_bytes()
    manifest = json.loads((first.run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluated"] is False
    assert manifest["candidate_count"] == 10
    assert manifest["completed_batches"] == 2
    assert manifest["status"] == "complete"


def test_exhaustion_is_explicit_in_manifest_and_export(tmp_path):
    config = ExpressionIterationConfig(
        search_space=SearchSpace(fields=("close",), operators=()),
        batch_size=2, max_attempts_per_batch=3, output_root=str(tmp_path),
    )
    result = run_expression_iteration(config)
    assert result.status == "exhausted" and result.candidate_count == 1
    manifest = json.loads((result.run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "exhausted"
    assert (result.run_path / "alphas.generated.json").is_file()


def test_failures_preserve_batch_records_and_mark_manifest(tmp_path):
    def fail(batch):
        raise RuntimeError("observer failed")

    config = ExpressionIterationConfig(batch_size=2, output_root=str(tmp_path))
    with pytest.raises(RuntimeError, match="observer failed"):
        run_expression_iteration(config, on_batch=fail)
    run_path, = tmp_path.iterdir()
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["candidate_count"] == 2
    assert len((run_path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_config_aliases_output_location_and_unknown_keys(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "expressions.json"
    path.write_text(json.dumps({
        "output_root": "../results", "search_space": {"operators": ["Mean", "Ref"]},
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    config = load_iteration_config(path)
    assert Path(config.output_root) == tmp_path / "results"
    assert config.search_space.operators == ("ts_mean", "ts_delay")
    path.write_text('{"batch_szie": 5}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid expression iteration config"):
        load_iteration_config(path)

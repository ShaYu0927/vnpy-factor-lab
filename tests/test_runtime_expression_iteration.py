from datetime import datetime, timedelta
import json

import polars as pl
import pytest

from vnpy import main
from vnpy.alpha.alpha import Alpha
from vnpy.alpha.engine import AlphaEngine, AlphaSampleCache
from vnpy.alpha.mining.runtime import prepare_runtime_alphas
from vnpy.config.runtime_config import ParquetConfig, RunMode, RuntimeConfig, load_runtime_config
from vnpy.datafeed.data_bar_cache import BarCache
from vnpy.datafeed.data_model import MarketBar
from vnpy.event.engine import ModuleEngine
from vnpy.event.event import EventType
from vnpy.factor.realtime_service import RealtimeAlphaService
from vnpy.strategy.strategy_engine import StrategyEngine


def runtime_config(tmp_path, *, enabled=True, fields=("volume",), operators=(), count=1):
    return RuntimeConfig(
        RunMode.PARQUET, ParquetConfig("unused", "2026-01-01", "2026-01-20"),
        raw={
            "alphas": [{"name": "fixed", "formula": "close"}],
            "expression_iteration": {
                "enabled": enabled, "batches": 1, "batch_size": count,
                "max_attempts_per_batch": 20, "output_root": "runs",
                "search_space": {"fields": fields, "operators": operators},
            },
        },
        config_dir=tmp_path,
    )


def test_runtime_generation_merges_in_memory_and_does_not_mutate_config(tmp_path):
    config = runtime_config(tmp_path)
    original = json.dumps(config.raw)
    resolved = prepare_runtime_alphas(config)
    assert [item.formula for item in resolved.definitions] == ["close", "volume"]
    assert resolved.generated_names == (resolved.definitions[1].name,)
    assert resolved.run_path.parent == tmp_path / "runs"
    assert json.dumps(config.raw) == original
    payload = json.loads((resolved.run_path / "alphas.runtime.json").read_text(encoding="utf-8"))
    assert payload["alphas"] == resolved.as_config()
    assert payload["optional_alphas"] == list(resolved.generated_names)


def test_disabled_generation_preserves_existing_flow(tmp_path):
    resolved = prepare_runtime_alphas(runtime_config(tmp_path, enabled=False))
    assert resolved.generated_names == () and resolved.run_path is None
    assert [item.name for item in resolved.definitions] == ["fixed"]
    assert not (tmp_path / "runs").exists()


def test_fixed_formula_keeps_its_name_when_generator_repeats_it(tmp_path):
    resolved = prepare_runtime_alphas(runtime_config(tmp_path, fields=("close",)))
    assert [item.name for item in resolved.definitions] == ["fixed"]
    assert not resolved.generated_names


def test_generation_exhaustion_does_not_silently_start_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "run_parquet_replay", lambda *a, **kw: pytest.fail("unexpected replay"))
    with pytest.raises(RuntimeError, match="exhausted"):
        main.run_from_config(runtime_config(tmp_path, count=2))


def test_runtime_loader_validates_inline_config_and_resolves_paths(tmp_path, monkeypatch):
    config = runtime_config(tmp_path)
    raw = {**config.raw, "mode": "parquet", "parquet": {"root": "unused", "start": "2026-01-01", "end": "2026-01-20"}}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    resolved = prepare_runtime_alphas(load_runtime_config(path))
    assert resolved.run_path.parent == tmp_path / "runs"
    raw["expression_iteration"]["enabled"] = "false"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_runtime_config(path)


def test_optional_candidates_do_not_block_fixed_values_or_warmup():
    definitions = [
        Alpha("fixed", "close / Ref(close, 1) - 1"),
        Alpha("invalid_candidate", "close / 0"),
        Alpha("slow_candidate", "Mean(close, 3)"),
    ]
    service = RealtimeAlphaService(
        BarCache(), AlphaSampleCache(), definitions, frequency="1d",
        universe=("A",),
        optional_names=("invalid_candidate", "slow_candidate"),
    )
    assert service.on_bar(MarketBar("B", datetime(2026, 1, 1), 10, 10, 10, 10, frequency="1d")) is None
    assert "B" not in service.bar_cache.symbols()
    samples = []
    for i in range(3):
        bar = MarketBar("A", datetime(2026, 1, 1) + timedelta(days=i), 10 + i, 10 + i, 10 + i, 10 + i, frequency="1d")
        samples.append(service.on_bar(bar))
    assert samples[0] is None
    assert samples[1].features == pytest.approx({"fixed": 0.1})
    assert samples[2].features == pytest.approx({"fixed": 12 / 11 - 1, "slow_candidate": 11.0})
    # Required-factor completeness is unchanged when no optional names are supplied.
    frame = service.alpha_engine.from_bars({"A": service.bar_cache.get_bars("A", frequency="1d")})
    assert service.alpha_engine.calculate_latest(frame) == []


def test_optional_only_empty_values_do_not_emit_empty_samples():
    service = RealtimeAlphaService(
        BarCache(), AlphaSampleCache(), [Alpha("bad", "close / 0")],
        optional_names=("bad",), frequency="1d",
    )
    service.on_bar(MarketBar("A", datetime(2026, 1, 1), 10, 10, 10, 10, frequency="1d"))
    assert not service.latest_samples


def test_main_replays_generated_values_to_real_strategy_engine_and_refreshes_modules(tmp_path, monkeypatch):
    folder = tmp_path / "source" / "SHSE_2026"
    folder.mkdir(parents=True)
    pl.DataFrame([
        {"trade_date": f"2026-01-{i + 1:02d}", "symbol": symbol,
         "open": 10.0 + i, "high": 12.0 + i, "low": 9.0 + i, "close": 11.0 + i,
         "volume": 100.0 + i, "amount": 1100.0 + i}
        for i in range(4) for symbol in ("SHSE.600000", "SHSE.600001")
    ]).write_parquet(folder / "dists_day_bar.parquet")
    config = runtime_config(tmp_path)
    config = RuntimeConfig(config.mode, ParquetConfig(
        str(folder.parent), "2026-01-01", "2026-01-04", symbols="SHSE.600000,SHSE.600001", markets="SHSE",
    ), config.raw, config.config_dir)
    monkeypatch.setattr(main, "module_engine", ModuleEngine())
    received = []
    original = StrategyEngine.on_event

    def observe(self, event):
        original(self, event)
        if event.event_type is EventType.FACTOR:
            received.append(event.get("sample"))

    monkeypatch.setattr(StrategyEngine, "on_event", observe)
    try:
        main.run_from_config(config)
        assert len(received) == 8
        assert len({(item.symbol, item.datetime) for item in received}) == 8
        for sample in received:
            generated, = set(sample.features) - {"fixed"}
            assert sample.features[generated] == 100 + sample.datetime.day - 1
        first_service = main.module_engine.get_context("factor").get_object("factor_service")

        # Running again in an IDE uses fresh formulas and caches, not the first run's service.
        received.clear()
        config.raw["expression_iteration"]["search_space"]["fields"] = ["high"]
        main.run_from_config(config)
        assert len(received) == 8
        assert first_service is not main.module_engine.get_context("factor").get_object("factor_service")
        for sample in received:
            generated, = set(sample.features) - {"fixed"}
            assert sample.features[generated] == 12 + sample.datetime.day - 1
    finally:
        main.module_engine.stop_all()


def test_import_branch_calculates_generated_and_fixed_columns(tmp_path):
    folder = tmp_path / "source" / "SHSE_2026"
    folder.mkdir(parents=True)
    pl.DataFrame([{
        "trade_date": "2026-01-01", "symbol": "SHSE.600000", "open": 10.0,
        "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0, "amount": 1100.0,
    }]).write_parquet(folder / "dists_day_bar.parquet")
    config = runtime_config(tmp_path)
    config.raw["parquet_import"] = {"enabled": True, "root": str(tmp_path / "imported")}
    config = RuntimeConfig(config.mode, ParquetConfig(
        str(folder.parent), "2026-01-01", "2026-01-01", symbols="SHSE.600000", markets="SHSE",
    ), config.raw, config.config_dir)
    main.run_from_config(config)
    output, = (tmp_path / "imported" / "imports").glob("*/factors.parquet")
    frame = pl.read_parquet(output)
    generated, = set(frame.columns) - {"datetime", "vt_symbol", "fixed"}
    assert frame[generated].to_list() == [100.0]
    assert frame["fixed"].to_list() == [11.0]

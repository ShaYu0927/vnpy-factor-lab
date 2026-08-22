import json

import pytest

from vnpy.config.runtime_config import RunMode, load_runtime_config


def test_load_gm_local_config(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "mode": "gm_local",
                "gm_local": {
                    "symbols": "SHSE.600519",
                    "frequency": "1d",
                    "count": 100,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.mode == RunMode.GM_LOCAL
    assert config.gm_local is not None
    assert config.gm_local.count == 100


def test_reject_missing_active_section(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text('{"mode": "database"}', encoding="utf-8")

    with pytest.raises(ValueError, match="database"):
        load_runtime_config(path)


def test_load_gm_sqlite_batch_config(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "mode": "gm_sqlite_batch",
                "gm_sqlite_batch": {
                    "root": "F:/Quantitative",
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                    "output": "data/factors.csv",
                    "batch_size": 5000,
                    "window_size": 21,
                    "max_workers": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.mode == RunMode.GM_SQLITE_BATCH
    assert config.gm_sqlite_batch is not None
    assert config.gm_sqlite_batch.batch_size == 5000
    assert config.gm_sqlite_batch.max_workers == 8
    assert config.gm_sqlite_batch.output == str(
        (tmp_path / "data" / "factors.csv").resolve()
    )


def test_reject_invalid_gm_sqlite_batch_size(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "mode": "gm_sqlite_batch",
                "gm_sqlite_batch": {
                    "root": "F:/Quantitative",
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                    "output": "data/factors.csv",
                    "batch_size": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size"):
        load_runtime_config(path)

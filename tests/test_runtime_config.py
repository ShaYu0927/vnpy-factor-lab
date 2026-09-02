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

import importlib.util
from pathlib import Path

import numpy as np


DEMO_PATH = Path(__file__).parents[1] / "examples" / "linear_regression_factor_demo.py"
SPEC = importlib.util.spec_from_file_location("linear_regression_factor_demo", DEMO_PATH)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_demo_runs_complete_price_to_prediction_flow() -> None:
    result = DEMO.run_demo()

    assert len(result["price_data"]) == 300
    assert len(result["factor_data"]) == 275
    assert len(result["predictions"]) == len(result["actual"])
    assert np.isfinite(result["predictions"]).all()
    assert np.isfinite(result["mse"])


def test_price_and_label_construction_is_reproducible() -> None:
    first = DEMO.make_price_data(seed=7)
    second = DEMO.make_price_data(seed=7)
    factor_data = DEMO.build_factor_data(first)

    assert first.equals(second)
    first_source_index = 20
    expected_label = first.loc[first_source_index + 5, "close"] / first.loc[first_source_index, "close"] - 1
    assert np.isclose(factor_data.loc[0, "label"], expected_label)

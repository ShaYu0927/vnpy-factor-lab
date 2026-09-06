from __future__ import annotations

import argparse

from .config import load_pipeline_config
from .workflow import AlphaTrainingPipeline, load_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a vn.py composable quant workflow")
    parser.add_argument("config", help="pipeline .json/.yaml config")
    parser.add_argument("--observations", required=True, help="factor observation JSON file")
    args = parser.parse_args()

    result = AlphaTrainingPipeline(load_pipeline_config(args.config)).run(
        load_observations(args.observations)
    )
    print(f"run_id={result.run_id}")
    print(f"run_path={result.run_path}")
    print(f"model_id={result.bundle.model_id}")


if __name__ == "__main__":
    main()

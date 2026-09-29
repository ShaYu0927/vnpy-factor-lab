"""Record expression-generation runs separately from scored factor mining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tempfile
from typing import Callable

import polars as pl

from .config import ExpressionIterationConfig
from .generator import iterate_expressions
from .schema import GenerationBatch
from vnpy.alpha.alpha import Alpha


@dataclass(frozen=True)
class IterationResult:
    run_path: Path
    status: str
    candidate_count: int
    alphas: tuple[Alpha, ...]


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_expression_iteration(
    config: ExpressionIterationConfig,
    on_batch: Callable[[GenerationBatch], None] | None = None,
) -> IterationResult:
    """Persist every batch; never replace a previous run or update runtime alphas."""
    root = Path(config.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("expressions_%Y%m%dT%H%M%SZ_")
    run_path = Path(tempfile.mkdtemp(prefix=stamp, dir=root))
    manifest = {
        "mode": "expression_generation", "generator_version": 1,
        "status": "running", "evaluated": False,
        "python_version": platform.python_version(), "polars_version": pl.__version__,
        "requested_candidates": config.batches * config.batch_size,
        "candidate_count": 0, "completed_batches": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_path / "config.json", asdict(config))
    _write_json(run_path / "manifest.json", manifest)
    alphas: list[dict[str, str]] = []
    definitions: list[Alpha] = []
    status = "complete"
    try:
        with (run_path / "candidates.jsonl").open("w", encoding="utf-8") as candidates_file, \
                (run_path / "batches.jsonl").open("w", encoding="utf-8") as batches_file:
            for batch in iterate_expressions(config):
                for candidate in batch.candidates:
                    candidates_file.write(json.dumps(candidate.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")
                    alpha = candidate.to_alpha()
                    definitions.append(alpha)
                    alphas.append({"name": alpha.name, "formula": alpha.formula})
                batches_file.write(json.dumps(batch.summary(), ensure_ascii=False) + "\n")
                candidates_file.flush()
                batches_file.flush()
                manifest["candidate_count"] = len(alphas)
                manifest["completed_batches"] += int(batch.complete)
                _write_json(run_path / "manifest.json", manifest)
                if on_batch is not None:
                    on_batch(batch)
                if not batch.complete:
                    status = "exhausted"
        # Compatible name/formula payload; these are unselected, unevaluated expressions.
        _write_json(run_path / "alphas.generated.json", {"alphas": alphas})
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = status
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_path / "manifest.json", manifest)
    return IterationResult(run_path, status, len(alphas), tuple(definitions))

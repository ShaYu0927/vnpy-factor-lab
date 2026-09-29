"""Resolve fixed and generated formulas for the application's existing factor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vnpy.alpha.alpha import Alpha
from vnpy.common.logger import get_logger
from vnpy.config.runtime_config import RuntimeConfig

from .config import runtime_iteration_config
from .schema import GenerationBatch, tree_fingerprint
from .workflow import _write_json, run_expression_iteration


@dataclass(frozen=True)
class RuntimeAlphas:
    definitions: tuple[Alpha, ...]
    generated_names: tuple[str, ...] = ()
    run_path: Path | None = None

    def as_config(self) -> list[dict[str, str]]:
        return [{"name": item.name, "formula": item.formula} for item in self.definitions]


def prepare_runtime_alphas(config: RuntimeConfig) -> RuntimeAlphas:
    """Generate once before data processing and pass the same formulas to import/replay."""
    raw = config.raw.get("alphas", [])
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError("alphas must be a list of name/formula objects")
    fixed = tuple(Alpha(**item) for item in raw)
    if len({item.name for item in fixed}) != len(fixed):
        raise ValueError("alpha names must be unique")
    # Validate fixed trees before producing any generation artifacts.
    from vnpy.alpha.engine import AlphaEngine
    AlphaEngine(fixed)
    options = runtime_iteration_config(config.raw.get("expression_iteration", {}), config.config_dir)
    if options is None:
        return RuntimeAlphas(fixed)

    logger = get_logger("main.expressions")

    def report(batch: GenerationBatch) -> None:
        logger.info("[expressions/batch] batch=%d generated=%d attempts=%d duplicates=%d rejected=%s",
                    batch.index, len(batch.candidates), batch.attempts, batch.duplicates, batch.rejected)

    result = run_expression_iteration(options, on_batch=report)
    if result.status != "complete":
        raise RuntimeError(f"expression generation {result.status}; partial results: {result.run_path}")
    fingerprints = {tree_fingerprint(item.parse()) for item in fixed}
    names = {item.name for item in fixed}
    generated = []
    for alpha in result.alphas:
        fingerprint = tree_fingerprint(alpha.parse())
        if fingerprint in fingerprints:
            continue
        if alpha.name in names:
            raise ValueError(f"generated alpha name conflicts with a fixed alpha: {alpha.name}")
        fingerprints.add(fingerprint)
        names.add(alpha.name)
        generated.append(alpha)
    resolved = RuntimeAlphas(fixed + tuple(generated), tuple(item.name for item in generated), result.run_path)
    _write_json(result.run_path / "alphas.runtime.json", {
        "alphas": resolved.as_config(), "optional_alphas": list(resolved.generated_names),
    })
    logger.info("[expressions/ready] fixed=%d generated=%d total=%d output=%s",
                len(fixed), len(generated), len(resolved.definitions), result.run_path)
    return resolved

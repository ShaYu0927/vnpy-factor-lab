from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping


class RunStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    run_id: str
    name: str
    path: str


class LocalRun:
    """A durable local experiment run with atomic metadata updates."""

    def __init__(self, root: Path, experiment: str, tags: Mapping[str, str]) -> None:
        timestamp = datetime.now(timezone.utc)
        self.run_id = f"{timestamp:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.path = root / experiment / self.run_id
        self.artifact_path = self.path / "artifacts"
        self.artifact_path.mkdir(parents=True, exist_ok=False)
        self._metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": experiment,
            "status": RunStatus.RUNNING.value,
            "started_at": timestamp.isoformat(),
            "finished_at": None,
            "tags": dict(tags),
            "params": {},
            "metrics": {},
            "error": None,
        }
        self._write_metadata()

    def __enter__(self) -> LocalRun:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exception is None:
            self.finish(RunStatus.FINISHED)
        else:
            self._metadata["error"] = f"{exception_type.__name__}: {exception}"
            self.finish(RunStatus.FAILED)
        return False

    def log_params(self, params: Mapping[str, object]) -> None:
        self._assert_running()
        self._metadata["params"].update(_json_safe(dict(params)))
        self._write_metadata()

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._assert_running()
        for name, value in metrics.items():
            self._metadata["metrics"][name] = float(value)
        self._write_metadata()

    def log_artifact(self, name: str, source: str | Path) -> ArtifactRef:
        self._assert_running()
        safe_name = _safe_artifact_name(name)
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target = self.artifact_path / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        return ArtifactRef(self.run_id, safe_name, str(target))

    def write_json_artifact(self, name: str, value: object) -> ArtifactRef:
        self._assert_running()
        safe_name = _safe_artifact_name(name)
        target = self.artifact_path / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return ArtifactRef(self.run_id, safe_name, str(target))

    def finish(self, status: RunStatus) -> None:
        if self._metadata["status"] != RunStatus.RUNNING.value:
            return
        self._metadata["status"] = status.value
        self._metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata()

    def _assert_running(self) -> None:
        if self._metadata["status"] != RunStatus.RUNNING.value:
            raise RuntimeError("experiment run is already closed")

    def _write_metadata(self) -> None:
        target = self.path / "run.json"
        temporary = self.path / ".run.json.tmp"
        temporary.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)


class LocalRecorder:
    def __init__(self, uri: str | Path = "./runs") -> None:
        self.root = Path(uri).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self, experiment: str, tags: Mapping[str, str] | None = None) -> LocalRun:
        if not experiment.strip() or Path(experiment).name != experiment:
            raise ValueError("experiment must be a non-empty path-safe name")
        return LocalRun(self.root, experiment, tags or {})


def _safe_artifact_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError("artifact name must be a relative path without '..'")
    return path.as_posix()


def _json_safe(value: object) -> Any:
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, Enum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)

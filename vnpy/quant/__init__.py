"""Composable research workflow built on top of the existing vn.py alpha stack."""

from .config import ComponentSpec, PipelineConfig, load_pipeline_config
from .experiment import LocalRecorder, RunStatus
from .model import ModelBundle, SignalFrame
from .registry import ComponentRegistry
from .workflow import AlphaTrainingPipeline, PipelineResult

__all__ = [
    "ComponentRegistry",
    "ComponentSpec",
    "AlphaTrainingPipeline",
    "LocalRecorder",
    "ModelBundle",
    "PipelineConfig",
    "PipelineResult",
    "RunStatus",
    "SignalFrame",
    "load_pipeline_config",
]

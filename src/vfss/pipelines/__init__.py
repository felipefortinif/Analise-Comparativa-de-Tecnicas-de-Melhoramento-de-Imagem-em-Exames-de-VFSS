"""Pipelines executáveis. O campo ``pipeline`` do experimento escolhe qual roda."""

from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig

from vfss.pipelines import extract_frames

PIPELINES: dict[str, Callable[[DictConfig, Path], None]] = {
    "extract_frames": extract_frames.run,
}

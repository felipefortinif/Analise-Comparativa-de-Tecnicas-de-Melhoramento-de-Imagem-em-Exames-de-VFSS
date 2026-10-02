"""Execução de um experimento: semente -> registro no MLflow -> pipeline."""

from pathlib import Path

from omegaconf import DictConfig

from vfss.pipelines import PIPELINES
from vfss.seed import seed_everything
from vfss.tracking import tracked_run


def execute(cfg: DictConfig, run_dir: Path) -> None:
    if cfg.pipeline not in PIPELINES:
        raise ValueError(f"Pipeline '{cfg.pipeline}' desconhecido. Disponíveis: {sorted(PIPELINES)}")
    seed_everything(cfg.seed, cfg.deterministic)
    with tracked_run(cfg, run_dir):
        PIPELINES[cfg.pipeline](cfg, run_dir)

"""Ponto de entrada único:  uv run python -m vfss experiment=<nome> [chave=valor ...]"""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from vfss.config import CONFIG_DIR
from vfss.runner import execute


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    execute(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()

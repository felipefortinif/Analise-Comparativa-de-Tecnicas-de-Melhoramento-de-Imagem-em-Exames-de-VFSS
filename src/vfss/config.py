"""Carregamento da configuração (Hydra + OmegaConf).

Todo parâmetro experimental vem de configs/. Este módulo só localiza os
arquivos e oferece `load_config` para testes e notebooks, que não passam
pela linha de comando.
"""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: PROJECT_ROOT.as_posix())


def load_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Compõe a configuração como a linha de comando faria.

    Exemplo: ``load_config(["experiment=fase0_extracao_quadros"])``.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))

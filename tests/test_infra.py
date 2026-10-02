"""Garantias da infraestrutura obrigatória da Fase 0."""

import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from vfss.config import CONFIG_DIR, load_config
from vfss.pipelines import PIPELINES
from vfss.seed import seed_everything
from vfss.video import to_grayscale

EXPERIMENTS = sorted(p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml"))


def _draw() -> tuple[float, np.ndarray]:
    return random.random(), np.random.rand(5)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experimento_completo_e_consistente(name):
    cfg = load_config([f"experiment={name}"])
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.experiment_name == name, "experiment_name deve ser igual ao nome do arquivo"
    assert cfg.pipeline in PIPELINES


def test_mesma_semente_reproduz_sequencia():
    seed_everything(123, deterministic=True)
    first = _draw()
    seed_everything(123, deterministic=True)
    second = _draw()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])


def test_sementes_diferentes_divergem():
    seed_everything(1, deterministic=True)
    first = _draw()
    seed_everything(2, deterministic=True)
    assert first[0] != _draw()[0]


def test_semente_torch():
    torch = pytest.importorskip("torch")
    seed_everything(7, deterministic=True)
    first = torch.rand(5)
    seed_everything(7, deterministic=True)
    assert torch.equal(first, torch.rand(5))


def test_cinza_sem_perda_quando_canais_iguais():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out, identical = to_grayscale(np.stack([gray] * 3, axis=-1))
    assert identical
    np.testing.assert_array_equal(out, gray)


def test_cinza_ponderado_quando_canais_diferem():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 2] = 100
    out, identical = to_grayscale(bgr)
    assert not identical
    assert out.shape == (2, 2)

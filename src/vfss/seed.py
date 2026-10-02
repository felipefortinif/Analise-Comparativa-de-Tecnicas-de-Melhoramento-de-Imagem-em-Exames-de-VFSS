"""Sementes aleatórias e determinismo."""

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool) -> None:
    """Fixa as sementes de `random`, NumPy e, se instalado, PyTorch.

    Com ``deterministic=True`` o PyTorch passa a usar apenas algoritmos
    determinísticos na GPU (um pouco mais lento, mas duas execuções com a
    mesma semente produzem o mesmo resultado).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # vale para subprocessos (ex.: workers)
    if deterministic:
        # Precisa estar definido antes da primeira chamada ao cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)

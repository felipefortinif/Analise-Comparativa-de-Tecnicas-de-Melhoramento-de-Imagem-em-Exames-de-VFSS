"""Registro automático de cada execução no MLflow local.

Toda execução feita por ``python -m vfss`` passa por `tracked_run`, que grava
sem intervenção manual:

- a configuração completa resolvida (como parâmetros e como config.yaml);
- a proveniência: commit do git, se havia alterações não commitadas (com o
  diff), hash do uv.lock, versão do Python, sistema e GPU;
- o tempo de parede da execução;
- o status final, inclusive FAILED quando o pipeline lança exceção, porque
  toda execução realizada entra no relato.

As métricas do próprio pipeline são registradas com ``mlflow.log_metrics``
dentro do bloco.
"""

import contextlib
import hashlib
import platform
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from vfss.config import PROJECT_ROOT


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout


def provenance() -> dict[str, str]:
    """Identifica exatamente o código e o ambiente que produziram um resultado."""
    lock = PROJECT_ROOT / "uv.lock"
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
    return {
        "git_commit": _run(["git", "rev-parse", "--verify", "-q", "HEAD"]).strip() or "sem-commit",
        "git_dirty": str(bool(_run(["git", "status", "--porcelain"]).strip())),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "ausente",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu or "nenhuma",
    }


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _experiment_id(name: str, artifact_root: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=Path(artifact_root).resolve().as_uri())


@contextlib.contextmanager
def tracked_run(cfg: DictConfig, run_dir: Path) -> Iterator[mlflow.ActiveRun]:
    """Abre uma execução no MLflow com configuração e proveniência já registradas."""
    uri = cfg.tracking.uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    experiment_id = _experiment_id(cfg.experiment_name, cfg.tracking.artifact_root)

    params = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    tags = provenance()
    run_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags({**tags, "pipeline": cfg.pipeline, "run_dir": run_dir.as_posix()})
        mlflow.log_params(_flatten(params))
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        if tags["git_dirty"] == "True":
            mlflow.log_text(_run(["git", "status", "--porcelain"]), "git_status.txt")
            mlflow.log_text(_run(["git", "diff", "HEAD"]), "git_diff.patch")
        hydra_dir = run_dir / ".hydra"  # inclui overrides.yaml: a linha de comando exata
        if hydra_dir.is_dir():
            mlflow.log_artifacts(str(hydra_dir), "hydra")
        (run_dir / "mlflow_run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        start = time.perf_counter()
        try:
            yield run
        finally:
            mlflow.log_metric("wall_time_s", time.perf_counter() - start)

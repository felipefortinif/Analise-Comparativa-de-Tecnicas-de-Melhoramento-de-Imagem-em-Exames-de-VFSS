"""Fase 0 — extrai os quadros de cada exame e grava no formato padronizado.

Estrutura de ``extract.output_dir``::

    <video_id>/<quadro com 6 dígitos>.png   PNG sem perda, 8 ou 16 bits
    frames.csv       uma linha por quadro gravado
    videos.csv       uma linha por exame lido com sucesso
    failures.csv     exames que não puderam ser lidos ou associados a um paciente
    provenance.json  execução do MLflow, commit e configuração que geraram a pasta

Use `read_extraction` para carregar os CSVs com os identificadores como texto
(um paciente "007" viraria o número 7 na leitura padrão do pandas).
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from vfss.tracking import provenance
from vfss.video import VideoEntry, discover_videos, open_video, to_grayscale

log = logging.getLogger(__name__)

FRAME_COLUMNS = [
    "video_id", "patient_id", "source", "frame_index", "time_s", "path",
    "height", "width", "channels", "dtype",
]
VIDEO_COLUMNS = [
    "video_id", "patient_id", "source", "video_name", "source_relpath", "source_sha256",
    "reader", "n_frames_declared", "n_frames_decoded", "n_frames_saved", "fps",
    "height", "width", "dtype", "bits_stored", "photometric", "channels_identical_frac",
]
FAILURE_COLUMNS = ["relpath", "stage", "error"]
ID_COLUMNS = {"video_id": str, "patient_id": str, "source": str, "video_name": str}
PROVENANCE_FILE = "provenance.json"


def run(cfg: DictConfig, run_dir: Path) -> None:
    c = cfg.extract
    input_dir = Path(c.input_dir)
    output_dir = Path(c.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Pasta de entrada não existe: {input_dir}")
    _prepare_output(output_dir, c.overwrite)

    discovery = discover_videos(
        input_dir, c.video_extensions, c.dicom_extensions,
        c.detect_extensionless_dicom, c.path_regex,
    )
    log.info("%d exames encontrados, %d rejeitados, %d arquivos ignorados",
             len(discovery.entries), len(discovery.rejected), discovery.n_ignored)

    frame_rows: list[dict] = []
    video_rows: list[dict] = []
    failures = list(discovery.rejected)
    for entry in discovery.entries:
        try:
            frames, video = _extract_video(entry, output_dir, c)
        except Exception as exc:
            if c.fail_fast:
                raise
            log.exception("Falha em %s", entry.relpath)
            failures.append({"relpath": entry.relpath, "stage": "leitura", "error": repr(exc)})
            shutil.rmtree(output_dir / entry.video_id, ignore_errors=True)
            continue
        frame_rows.extend(frames)
        video_rows.append(video)
        log.info("%s: %d quadros", entry.relpath, video["n_frames_saved"])

    paths = {
        "frames.csv": pd.DataFrame(frame_rows, columns=FRAME_COLUMNS),
        "videos.csv": pd.DataFrame(video_rows, columns=VIDEO_COLUMNS),
        "failures.csv": pd.DataFrame(failures, columns=FAILURE_COLUMNS),
    }
    for name, table in paths.items():
        table.to_csv(output_dir / name, index=False)
        mlflow.log_artifact(str(output_dir / name), "extraction")

    active = mlflow.active_run()
    (output_dir / PROVENANCE_FILE).write_text(json.dumps({
        "mlflow_run_id": active.info.run_id if active else None,
        "run_dir": run_dir.as_posix(),
        **provenance(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    videos = paths["videos.csv"]
    mlflow.log_metrics({
        "n_videos": len(videos),
        "n_patients": videos["patient_id"].nunique(),
        "n_sources": videos["source"].nunique(),
        "n_frames_saved": len(frame_rows),
        "n_failures": len(failures),
        "n_ignored_files": discovery.n_ignored,
    })
    log.info("Extração concluída em %s: %d exames, %d quadros, %d falhas",
             output_dir, len(videos), len(frame_rows), len(failures))


def read_extraction(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega (frames, videos, failures) com os identificadores como texto."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(output_dir / name, dtype=ID_COLUMNS, keep_default_na=False,
                           na_values=[""])

    return read("frames.csv"), read("videos.csv"), read("failures.csv")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} já contém uma extração. Use outro experiment_name "
                "ou extract.overwrite=true."
            )
        # Só apaga pastas criadas por este pipeline, nunca um caminho qualquer.
        if not (output_dir / PROVENANCE_FILE).is_file():
            raise FileExistsError(f"{output_dir} não foi criada por este pipeline; não vou apagá-la.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _extract_video(entry: VideoEntry, output_dir: Path, c: DictConfig) -> tuple[list[dict], dict]:
    meta, frames = open_video(entry)
    video_dir = output_dir / entry.video_id
    video_dir.mkdir()

    rows: list[dict] = []
    n_decoded = n_identical = 0
    for index, frame in enumerate(frames):
        n_decoded += 1
        if index % c.frame_step:
            continue
        if c.max_frames_per_video is not None and len(rows) >= c.max_frames_per_video:
            continue
        if c.grayscale:
            frame, identical = to_grayscale(frame)
            n_identical += identical
        relpath = f"{entry.video_id}/{index:06d}.png"
        _write_png(output_dir / relpath, frame, c.png_compression)
        rows.append({
            "video_id": entry.video_id,
            "patient_id": entry.patient_id,
            "source": entry.source,
            "frame_index": index,
            "time_s": index / meta.fps if meta.fps else np.nan,
            "path": relpath,
            "height": frame.shape[0],
            "width": frame.shape[1],
            "channels": 1 if frame.ndim == 2 else frame.shape[2],
            "dtype": str(frame.dtype),
        })
    if not rows:
        raise ValueError(f"nenhum quadro gravado ({n_decoded} decodificados)")

    video = {
        "video_id": entry.video_id,
        "patient_id": entry.patient_id,
        "source": entry.source,
        "video_name": entry.video_name,
        "source_relpath": entry.relpath,
        "source_sha256": _sha256(entry.path) if c.hash_source_files else "",
        "reader": meta.reader,
        "n_frames_declared": meta.n_frames_declared,
        "n_frames_decoded": n_decoded,
        "n_frames_saved": len(rows),
        "fps": meta.fps,
        "height": meta.height,
        "width": meta.width,
        "dtype": rows[0]["dtype"],
        "bits_stored": meta.bits_stored,
        "photometric": meta.photometric,
        "channels_identical_frac": n_identical / len(rows) if c.grayscale else np.nan,
    }
    return rows, video


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _write_png(path: Path, frame: np.ndarray, compression: int) -> None:
    if frame.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"PNG aceita uint8 ou uint16, recebeu {frame.dtype}")
    ok, buffer = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise OSError(f"Falha ao codificar {path.name}")
    # cv2.imwrite falha em silêncio com caminhos acentuados no Windows.
    path.write_bytes(buffer.tobytes())

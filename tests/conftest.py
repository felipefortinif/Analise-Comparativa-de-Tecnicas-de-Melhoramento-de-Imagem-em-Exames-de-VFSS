"""Exames sintéticos para testar o pipeline sem dados clínicos."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import DictConfig
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from vfss.config import load_config

XRF_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.12.2"  # X-Ray Radiofluoroscopic Image


def write_dicom(path: Path, frames: np.ndarray, bits_stored: int, fps: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = XRF_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = XRF_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.set_pixel_data(frames, photometric_interpretation="MONOCHROME2", bits_stored=bits_stored)
    ds.CineRate = fps
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def write_avi(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height),
                             isColor=False)
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def raw_dataset(tmp_path: Path) -> dict:
    """Base sintética com duas origens, casos válidos e casos que devem falhar."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    dicom_12bit = rng.integers(0, 2**12, size=(5, 32, 48), dtype=np.uint16)

    write_avi(raw / "instA/pac001/deglut_1.avi",
              rng.integers(0, 256, size=(6, 32, 48), dtype=np.uint8), fps=30)
    write_dicom(raw / "instA/pac001/deglut_2.dcm", dicom_12bit, bits_stored=12, fps=15)
    write_dicom(raw / "instB/007/EXAME", rng.integers(0, 256, size=(4, 32, 48), dtype=np.uint8),
                bits_stored=8, fps=30)  # DICOM sem extensão; paciente com zero à esquerda
    write_avi(raw / "solto.avi", np.zeros((2, 32, 48), np.uint8), fps=30)  # fora do padrão
    (raw / "instB/pac003").mkdir(parents=True)
    (raw / "instB/pac003/corrompido.dcm").write_bytes(b"isto nao e um dicom")
    (raw / "instA/leia-me.txt").write_text("ignorado")

    return {"raw": raw, "dicom_12bit": dicom_12bit}


@pytest.fixture
def extract_cfg(tmp_path: Path, raw_dataset: dict) -> DictConfig:
    """Configuração do experimento real, redirecionada para pastas temporárias."""
    cfg = load_config(["experiment=fase0_extracao_quadros"])
    cfg.extract.input_dir = raw_dataset["raw"].as_posix()
    cfg.extract.output_dir = (tmp_path / "frames").as_posix()
    cfg.tracking.uri = "sqlite:///" + (tmp_path / "mlruns/mlflow.db").as_posix()
    cfg.tracking.artifact_root = (tmp_path / "mlruns/artifacts").as_posix()
    return cfg

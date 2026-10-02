"""Descoberta e leitura de exames de VFSS: vídeo comum (OpenCV) ou DICOM (pydicom).

Os quadros saem exatamente como estão armazenados, sem nenhum ajuste de
intensidade. Qualquer transformação é uma etapa posterior e configurada.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixels import iter_pixels


@dataclass(frozen=True)
class VideoEntry:
    path: Path
    relpath: str  # relativo à pasta de entrada, sempre com "/"
    kind: str  # "dicom" | "video"
    source: str
    patient_id: str
    video_name: str

    @property
    def video_id(self) -> str:
        raw = f"{self.source}__{self.patient_id}__{self.video_name}"
        return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)


@dataclass
class Discovery:
    entries: list[VideoEntry] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    n_ignored: int = 0  # arquivos que não são vídeo nem DICOM


@dataclass(frozen=True)
class VideoMeta:
    reader: str
    height: int
    width: int
    fps: float | None
    n_frames_declared: int | None
    # Profundidade entregue ao pipeline. No DICOM é o BitsStored original; o
    # OpenCV sempre decodifica em 8 bits, mesmo que a fonte tenha mais.
    bits_stored: int
    photometric: str | None


def is_dicom(path: Path) -> bool:
    with path.open("rb") as f:
        f.seek(128)
        return f.read(4) == b"DICM"


def discover_videos(
    input_dir: Path,
    video_extensions: Iterable[str],
    dicom_extensions: Iterable[str],
    detect_extensionless_dicom: bool,
    path_regex: str,
) -> Discovery:
    """Lista os exames em ``input_dir`` e associa cada um a origem e paciente.

    O paciente vem exclusivamente de ``path_regex`` (grupo nomeado
    ``patient_id``; ``source`` e ``video_name`` são opcionais). Arquivo que
    não casa com o padrão é rejeitado, nunca associado por palpite: a divisão
    dos dados por paciente depende disso.
    """
    pattern = re.compile(path_regex)
    if "patient_id" not in pattern.groupindex:
        raise ValueError("path_regex precisa de um grupo nomeado (?P<patient_id>...)")
    video_exts = {e.lower() for e in video_extensions}
    dicom_exts = {e.lower() for e in dicom_extensions}

    discovery = Discovery()
    seen_ids: dict[str, str] = {}
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix in dicom_exts or (detect_extensionless_dicom and not suffix and is_dicom(path)):
            kind = "dicom"
        elif suffix in video_exts:
            kind = "video"
        else:
            discovery.n_ignored += 1
            continue

        relpath = path.relative_to(input_dir).as_posix()
        match = pattern.search(relpath)
        if match is None:
            discovery.rejected.append(
                {"relpath": relpath, "stage": "descoberta", "error": "caminho fora de path_regex"}
            )
            continue
        groups = match.groupdict()
        entry = VideoEntry(
            path=path,
            relpath=relpath,
            kind=kind,
            source=groups.get("source") or "desconhecida",
            patient_id=groups["patient_id"],
            video_name=groups.get("video_name") or path.stem,
        )
        if entry.video_id in seen_ids:
            discovery.rejected.append({
                "relpath": relpath,
                "stage": "descoberta",
                "error": f"video_id repetido com {seen_ids[entry.video_id]}",
            })
            continue
        seen_ids[entry.video_id] = relpath
        discovery.entries.append(entry)
    return discovery


def open_video(entry: VideoEntry) -> tuple[VideoMeta, Iterator[np.ndarray]]:
    """Devolve os metadados e um iterador preguiçoso sobre os quadros.

    Quadros com 3 canais saem sempre em ordem BGR, como no OpenCV.
    """
    if entry.kind == "dicom":
        return _open_dicom(entry.path)
    return _open_opencv(entry.path)


def _open_dicom(path: Path) -> tuple[VideoMeta, Iterator[np.ndarray]]:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    if ds.get("PixelRepresentation", 0) != 0:
        raise ValueError("DICOM com pixels com sinal ainda não é suportado")

    fps = None
    for keyword in ("CineRate", "RecommendedDisplayFrameRate"):
        if ds.get(keyword):
            fps = float(ds.get(keyword))
            break
    if fps is None and ds.get("FrameTime"):
        fps = 1000.0 / float(ds.FrameTime)  # FrameTime é em ms

    meta = VideoMeta(
        reader="dicom",
        height=int(ds.Rows),
        width=int(ds.Columns),
        fps=fps,
        n_frames_declared=int(ds.get("NumberOfFrames", 1)),
        bits_stored=int(ds.BitsStored),
        photometric=str(ds.PhotometricInterpretation),
    )
    rgb = int(ds.get("SamplesPerPixel", 1)) == 3

    def frames() -> Iterator[np.ndarray]:
        for frame in iter_pixels(path):
            yield frame[..., ::-1] if rgb else frame

    return meta, frames()


def _open_opencv(path: Path) -> tuple[VideoMeta, Iterator[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(
            f"OpenCV não conseguiu abrir {path.name} "
            "(codec sem suporte ou caminho com acentos no Windows)"
        )
    fps = capture.get(cv2.CAP_PROP_FPS)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = VideoMeta(
        reader="opencv",
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        fps=fps if fps > 0 else None,
        n_frames_declared=declared if declared > 0 else None,
        bits_stored=8,
        photometric=None,
    )

    def frames() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame
        finally:
            capture.release()

    return meta, frames()


def to_grayscale(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Converte para um canal e informa se os canais eram idênticos.

    Canais idênticos: copia um deles, sem perda. Diferentes (artefato de
    compressão de cor): conversão ponderada BGR -> cinza.
    """
    if frame.ndim == 2:
        return frame, True
    if frame.shape[2] == 1:
        return np.ascontiguousarray(frame[..., 0]), True
    b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
    if np.array_equal(b, g) and np.array_equal(b, r):
        return np.ascontiguousarray(b), True
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), False

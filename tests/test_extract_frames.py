import subprocess
import sys
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pytest

from vfss.config import PROJECT_ROOT
from vfss.pipelines.extract_frames import read_extraction
from vfss.runner import execute


def _runs(cfg) -> list:
    client = mlflow.MlflowClient(tracking_uri=cfg.tracking.uri)
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    return client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])


def test_extracao_completa(tmp_path, raw_dataset, extract_cfg):
    execute(extract_cfg, tmp_path / "run")
    out = Path(extract_cfg.extract.output_dir)
    frames, videos, failures = read_extraction(out)

    assert sorted(videos["patient_id"]) == ["007", "pac001", "pac001"]
    assert sorted(videos["source"].unique()) == ["instA", "instB"]
    assert sorted(failures["relpath"]) == ["instB/pac003/corrompido.dcm", "solto.avi"]
    assert len(frames) == 6 + 5 + 4
    assert (out / "provenance.json").is_file()
    assert not (out / "instB__pac003__corrompido").exists()  # quadros parciais removidos

    dicom = videos.set_index("video_id").loc["instA__pac001__deglut_2"]
    assert dicom["bits_stored"] == 12 and dicom["fps"] == 15 and dicom["dtype"] == "uint16"
    assert len(dicom["source_sha256"]) == 64


def test_dicom_12_bits_sai_identico(tmp_path, raw_dataset, extract_cfg):
    execute(extract_cfg, tmp_path / "run")
    out = Path(extract_cfg.extract.output_dir)
    frames, _, _ = read_extraction(out)
    rows = frames[frames["video_id"] == "instA__pac001__deglut_2"].sort_values("frame_index")

    for expected, rel in zip(raw_dataset["dicom_12bit"], rows["path"], strict=True):
        png = cv2.imdecode(np.fromfile(out / rel, np.uint8), cv2.IMREAD_UNCHANGED)
        assert png.dtype == np.uint16
        np.testing.assert_array_equal(png, expected)


def test_metricas_e_proveniencia_registradas(tmp_path, raw_dataset, extract_cfg):
    execute(extract_cfg, tmp_path / "run")
    [run] = _runs(extract_cfg)

    assert run.info.status == "FINISHED"
    assert run.data.metrics["n_frames_saved"] == 15
    assert run.data.metrics["n_patients"] == 2
    assert run.data.metrics["n_failures"] == 2
    assert "wall_time_s" in run.data.metrics
    assert run.data.params["seed"] == "42"
    assert run.data.params["extract.frame_step"] == "1"
    for tag in ("git_commit", "git_dirty", "uv_lock_sha256", "python", "gpu"):
        assert tag in run.data.tags
    assert (tmp_path / "run/mlflow_run_id.txt").read_text() == run.info.run_id


def test_nao_sobrescreve_e_registra_execucao_que_falhou(tmp_path, raw_dataset, extract_cfg):
    execute(extract_cfg, tmp_path / "run1")
    with pytest.raises(FileExistsError):
        execute(extract_cfg, tmp_path / "run2")

    assert [r.info.status for r in _runs(extract_cfg)] == ["FINISHED", "FAILED"]


def test_overwrite_substitui_extracao_anterior(tmp_path, raw_dataset, extract_cfg):
    execute(extract_cfg, tmp_path / "run1")
    extract_cfg.extract.overwrite = True
    extract_cfg.extract.max_frames_per_video = 2
    execute(extract_cfg, tmp_path / "run2")

    frames, _, _ = read_extraction(Path(extract_cfg.extract.output_dir))
    assert len(frames) == 3 * 2


def test_overwrite_recusa_pasta_que_nao_e_do_pipeline(tmp_path, raw_dataset, extract_cfg):
    out = Path(extract_cfg.extract.output_dir)
    out.mkdir()
    (out / "dados_importantes.txt").write_text("não apagar")
    extract_cfg.extract.overwrite = True

    with pytest.raises(FileExistsError):
        execute(extract_cfg, tmp_path / "run")
    assert (out / "dados_importantes.txt").exists()


def test_linha_de_comando(tmp_path, raw_dataset):
    """O caminho de uso real: python -m vfss experiment=... com overrides."""
    overrides = {
        "extract.input_dir": raw_dataset["raw"].as_posix(),
        "extract.output_dir": (tmp_path / "frames").as_posix(),
        "tracking.uri": "sqlite:///" + (tmp_path / "mlruns/mlflow.db").as_posix(),
        "tracking.artifact_root": (tmp_path / "mlruns/artifacts").as_posix(),
        "paths.runs": (tmp_path / "outputs").as_posix(),
    }
    args = ["experiment=fase0_extracao_quadros", *(f"{k}='{v}'" for k, v in overrides.items())]
    result = subprocess.run([sys.executable, "-m", "vfss", *args], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "frames/frames.csv").is_file()
    [run_dir] = (tmp_path / "outputs/fase0_extracao_quadros").iterdir()
    assert (run_dir / ".hydra/overrides.yaml").is_file()
    assert (run_dir / "mlflow_run_id.txt").is_file()

import pytest

from atlas_training import config as cfg
from atlas_training.train import base_weights


def test_bare_name_is_an_ultralytics_model_in_weights_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WEIGHTS_DIR", tmp_path / "weights")
    assert base_weights("yolo11s.pt") == tmp_path / "weights" / "yolo11s.pt"


def test_path_is_an_earlier_run_relative_to_training_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "TRAINING_DIR", tmp_path)
    best = tmp_path / "runs" / "train" / "v3" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.touch()
    assert base_weights("runs/train/v3/weights/best.pt") == best


def test_missing_earlier_run_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "TRAINING_DIR", tmp_path)
    with pytest.raises(SystemExit):
        base_weights("runs/train/missing/weights/best.pt")

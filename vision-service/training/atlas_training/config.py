"""Experiment configuration (configs/*.yaml) and the folders used for large files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# vision-service/training/
TRAINING_DIR = Path(__file__).resolve().parent.parent
# Large files stay out of Git (see .gitignore): downloads, prepared datasets,
# weights and run outputs.
DATA_DIR = TRAINING_DIR / "datasets"
DOWNLOAD_DIR = DATA_DIR / "raw"
WEIGHTS_DIR = TRAINING_DIR / "weights"
RUNS_DIR = TRAINING_DIR / "runs"

IGNORE = "ignore"
DROP_IMAGE = "drop_image"

MAX_STEM = 80


def short_stem(stem: str) -> str:
    """Keep file names short enough for Windows' 260-character path limit.

    Stems longer than MAX_STEM become their first 40 characters plus 12 characters of
    their SHA-1, so the same stem always gets the same short name.
    """
    if len(stem) <= MAX_STEM:
        return stem
    return f"{stem[:40]}_{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True)
class DatasetSource:
    id: str
    workspace: str
    project: str
    version: int | None
    licence: str
    url: str
    class_map: dict[str, str]
    move_to_test: float = 0.0
    upright_test_copies: bool = False
    clip_similarity: float | None = None
    orientation_overrides: dict[str, int] = field(default_factory=dict)
    augmented_copies: bool = False
    exclude_images: dict[str, str] = field(default_factory=dict)  # file stem: reason
    train_repeat: int = 1                     # each train image is written this many times
    repeat_background: bool = True            # False: train_repeat only for images with a target box
    max_train_background: int | None = None   # cap on train images without a target box

    @property
    def download_dir(self) -> Path:
        return DOWNLOAD_DIR / f"{self.id}-v{self.version}"


@dataclass(frozen=True)
class PretrainedModel:
    id: str
    url: str
    licence: str
    ambulance_class: str

    @property
    def path(self) -> Path:
        return WEIGHTS_DIR / f"{self.id}.pt"


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    target_classes: list[str]
    datasets: list[DatasetSource]
    test_fraction: float
    val_fraction: float
    pretrained: list[PretrainedModel]
    train: dict[str, Any] = field(default_factory=dict)
    iou: float = 0.5
    conf: float = 0.25
    test_near_copy_similarity: float | None = None

    @property
    def dataset_dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"


def load(path: str | Path) -> Experiment:
    """Read and check an experiment file. Errors name the field that is wrong."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    targets = list(raw["target_classes"])
    allowed = set(targets) | {IGNORE, DROP_IMAGE}

    sources = []
    for d in raw["datasets"]:
        class_map = {str(k): str(v) for k, v in d["class_map"].items()}
        bad = {k: v for k, v in class_map.items() if v not in allowed}
        if bad:
            raise ValueError(f"{d['id']}: class_map values must be one of {sorted(allowed)}, got {bad}")
        move_to_test = float(d.get("move_to_test", 0.0))
        if not 0 <= move_to_test < 0.5:
            raise ValueError(f"{d['id']}: move_to_test must be between 0 and 0.5")
        clip_similarity = d.get("clip_similarity")
        if clip_similarity is not None and not 0 < float(clip_similarity) < 1:
            raise ValueError(f"{d['id']}: clip_similarity must be between 0 and 1")
        exclude = {str(k): str(v) for k, v in (d.get("exclude_images") or {}).items()}
        train_repeat = int(d.get("train_repeat", 1))
        if not 1 <= train_repeat <= 10:
            raise ValueError(f"{d['id']}: train_repeat must be between 1 and 10")
        max_background = d.get("max_train_background")
        if max_background is not None and int(max_background) < 0:
            raise ValueError(f"{d['id']}: max_train_background must not be negative")
        overrides = {str(k): int(v) % 4 for k, v in (d.get("orientation_overrides") or {}).items()}
        sources.append(DatasetSource(
            id=d["id"], workspace=d["workspace"], project=d["project"],
            version=d.get("version"), licence=d["licence"], url=d["url"],
            class_map=class_map, move_to_test=move_to_test,
            upright_test_copies=bool(d.get("upright_test_copies", False)),
            clip_similarity=None if clip_similarity is None else float(clip_similarity),
            orientation_overrides=overrides,
            augmented_copies=bool(d.get("augmented_copies", False)),
            exclude_images=exclude,
            train_repeat=train_repeat,
            repeat_background=bool(d.get("repeat_background", True)),
            max_train_background=None if max_background is None else int(max_background),
        ))

    split = raw["split"]
    for key in ("test_fraction", "val_fraction"):
        if not 0 < split[key] < 0.5:
            raise ValueError(f"split.{key} must be between 0 and 0.5")
    near_copy = split.get("drop_test_near_copies")
    if near_copy is not None and not 0 < float(near_copy) < 1:
        raise ValueError("split.drop_test_near_copies must be between 0 and 1")

    evaluate = raw.get("evaluate", {})
    return Experiment(
        name=raw["name"],
        seed=int(raw["seed"]),
        target_classes=targets,
        datasets=sources,
        test_fraction=float(split["test_fraction"]),
        val_fraction=float(split["val_fraction"]),
        pretrained=[PretrainedModel(**p) for p in raw.get("pretrained", [])],
        train=dict(raw.get("train", {})),
        iou=float(evaluate.get("iou", 0.5)),
        conf=float(evaluate.get("conf", 0.25)),
        test_near_copy_similarity=None if near_copy is None else float(near_copy),
    )

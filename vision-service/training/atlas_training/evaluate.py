"""Compare models on the held-out test set of an experiment.

    python -m atlas_training.evaluate configs/ambulance_v1.yaml
    python -m atlas_training.evaluate configs/ambulance_v1.yaml --model finetuned=runs/train/ambulance_v1/weights/best.pt

Without --model, the pre-trained models of the config are evaluated; --model adds
more (id=path, repeatable). Every model is reduced to its ambulance boxes and scored
with atlas_training.metrics, so models with different class lists are compared on the
same images in the same way. Results go to runs/eval/<experiment>/: one JSON file per
model and results.md with the comparison against the NFR-04 target.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import config as cfg
from .metrics import ImageResult, Scores, best_f1_threshold, score
from .prepare import IMAGE_SUFFIXES

PREDICT_CHUNK = 16  # images per predict call

NFR04_TARGET = 0.90


def load_ground_truth(label_file: Path, width: int, height: int, cls: int = 0) -> np.ndarray:
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) == cls:
                cx, cy, w, h = (float(v) for v in parts[1:5])
                boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                              (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def class_index(names: dict[int, str], wanted: str) -> int:
    for idx, name in names.items():
        if name.lower() == wanted.lower():
            return int(idx)
    raise SystemExit(f"class {wanted!r} not found in the model's classes {list(names.values())}")


def run_model(weights: Path, class_name: str, exp: cfg.Experiment) -> list[ImageResult]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    idx = class_index(model.names, class_name)
    test_dir = exp.dataset_dir / "test"
    images = sorted(p for p in (test_dir / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    results = []
    # A low threshold keeps the whole precision-recall curve for AP; the reported
    # precision and recall use exp.conf.
    # Ultralytics runs a list source as one batch, which does not fit in GPU memory
    # for a whole test set, so the images go in chunks.
    predictions = (
        pred
        for start in range(0, len(images), PREDICT_CHUNK)
        for pred in model.predict(
            [str(p) for p in images[start:start + PREDICT_CHUNK]], conf=0.001, classes=[idx],
            imgsz=exp.train.get("imgsz", 640), verbose=False,
        )
    )
    for image, pred in zip(images, predictions, strict=True):
        height, width = pred.orig_shape
        gt = load_ground_truth(test_dir / "labels" / f"{image.stem}.txt", width, height)
        boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes is not None else np.zeros((0, 4))
        scores = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.zeros(0)
        results.append(ImageResult(gt=gt, boxes=boxes, scores=scores))
    return results


def gap(value: float) -> str:
    diff = value - NFR04_TARGET
    return "meets" if diff >= 0 else f"{diff * 100:+.1f} pt"


def report(rows: dict[str, dict], exp: cfg.Experiment) -> str:
    lines = [
        f"# {exp.name}: ambulance detection on the held-out test set",
        "",
        (f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. IoU >= {exp.iou}; precision and "
         f"recall at confidence >= {exp.conf}. NFR-04 target: precision and recall >= {NFR04_TARGET:.0%}."),
        "",
        "| Model | Images | Ambulance boxes | Precision | Recall | F1 | AP@0.5 | Precision vs NFR-04 | Recall vs NFR-04 | Best-F1 threshold (P / R) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_id, r in rows.items():
        s, b = r["at_conf"], r["best_f1"]
        best = f"{b['conf']:.2f} ({b['precision']:.3f} / {b['recall']:.3f})" if b["f1"] > 0 else "–"
        lines.append(
            f"| {model_id} | {s['n_images']} | {s['n_gt']} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['ap50']:.3f} | {gap(s['precision'])} | {gap(s['recall'])} | {best} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--model", action="append", default=[], metavar="ID=PATH",
                        help="extra model whose ambulance class is called 'ambulance'")
    args = parser.parse_args()
    exp = cfg.load(args.config)
    if not (exp.dataset_dir / "test" / "images").is_dir():
        raise SystemExit("Prepare the dataset first: python -m atlas_training.prepare <config>")

    models = {m.id: (m.path, m.ambulance_class) for m in exp.pretrained}
    for item in args.model:
        model_id, _, path = item.partition("=")
        models[model_id] = (Path(path), "ambulance")

    out = cfg.RUNS_DIR / "eval" / exp.name
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for model_id, (path, class_name) in models.items():
        if not path.exists():
            print(f"{model_id}: {path} not found, skipped")
            continue
        print(f"{model_id}: evaluating {path}")
        images = run_model(path, class_name, exp)
        at_conf: Scores = score(images, exp.iou, exp.conf)
        best: Scores = best_f1_threshold(images, exp.iou)
        rows[model_id] = {"weights": str(path), "at_conf": at_conf.as_dict(), "best_f1": best.as_dict()}
        (out / f"{model_id}.json").write_text(json.dumps(rows[model_id], indent=2) + "\n")
    text = report(rows, exp)
    (out / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

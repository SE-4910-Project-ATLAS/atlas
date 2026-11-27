"""Two-stage ambulance detection: COCO vehicle detector, then an ambulance classifier on each crop.

    python -m atlas_training.twostage configs/ambulance_v5.yaml \
        --classifier runs/cls/crops_v5/weights/best.pt --compare v3=runs/train/ambulance_v3/weights/best.pt@640

Each vehicle box (COCO car, bus, truck) with detector confidence >= --det-conf is cropped
like atlas_training.crops does for training and classified; the box is an ambulance
detection with score p(ambulance). Scored on the held-out test split with
atlas_training.metrics, so the numbers compare directly with atlas_training.evaluate, and
broken down by source and by video of our own frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .evaluate import class_index, load_ground_truth
from .metrics import ImageResult, best_f1_threshold, score
from .prepare import IMAGE_SUFFIXES


class TwoStage:
    def __init__(self, detector: Path, classifier: Path, det_conf: float, imgsz: int, pad: float, size: int):
        from ultralytics import YOLO

        self.detector, self.classifier = YOLO(str(detector)), YOLO(str(classifier))
        self.amb = class_index(self.classifier.names, "ambulance")
        self.det_conf, self.imgsz, self.pad, self.size = det_conf, imgsz, pad, size

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        r = self.detector.predict(image, conf=self.det_conf, imgsz=self.imgsz, classes=VEHICLE_CLASSES,
                                   agnostic_nms=True, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        crops = [cv2.resize(square_crop(image, b, self.pad), (self.size, self.size)) for b in boxes]
        probs = [p.probs.data[self.amb].item() for p in self.classifier.predict(crops, imgsz=self.size, verbose=False)]
        return boxes, np.asarray(probs)


def one_stage(weights: Path, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    idx = class_index(model.names, "ambulance")

    def run(image):
        r = model.predict(image, conf=0.001, classes=[idx], imgsz=imgsz, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return run


def fuse(one: ImageResult, two: ImageResult, min_two: float, iou: float = 0.5) -> ImageResult:
    """One-stage boxes as they are, plus two-stage boxes that no one-stage box covers and
    whose classifier probability is at least min_two. A covered two-stage box raises the
    score of the one-stage box it overlaps to the higher of the two."""
    from .metrics import box_iou

    boxes, scores = list(one.boxes), list(one.scores)
    for b, s in zip(two.boxes, two.scores, strict=True):
        if len(one.boxes) and (overlap := box_iou(b[None], one.boxes)[0]).max() >= iou:
            j = int(overlap.argmax())
            scores[j] = max(scores[j], s)
        elif s >= min_two:
            boxes.append(b)
            scores.append(s)
    return ImageResult(gt=one.gt, boxes=np.asarray(boxes, float).reshape(-1, 4), scores=np.asarray(scores, float))


def breakdown(names: list[str], results: list[ImageResult], iou: float, conf: float) -> dict:
    out = {}
    groups: dict[str, list[ImageResult]] = {}
    for n, r in zip(names, results, strict=True):
        source, _, stem = n.partition("__")
        groups.setdefault(source, []).append(r)
        if source == "own_cctv":
            groups.setdefault("own_cctv/" + Path(stem).stem.rsplit("_f", 1)[0], []).append(r)
    for g, rs in sorted(groups.items()):
        s = score(rs, iou, conf)
        out[g] = {"gt": s.n_gt, "tp": s.tp, "fp": s.fp, "precision": round(s.precision, 3), "recall": round(s.recall, 3), "ap50": round(s.ap50, 3)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--compare", action="append", default=[], metavar="ID=WEIGHTS@IMGSZ")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import cv2

    exp = cfg.load(args.config)
    test = exp.dataset_dir / "test"
    images = sorted(p for p in (test / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    models = {"two_stage": TwoStage(args.detector, args.classifier, args.det_conf, args.imgsz, args.pad, args.size)}
    for item in args.compare:
        model_id, _, rest = item.partition("=")
        path, _, size = rest.partition("@")
        models[model_id] = one_stage(Path(path), int(size or 640))
    report, raw = {}, {}
    for model_id, run in models.items():
        results = []
        raw[model_id] = results
        for p in images:
            image = cv2.imread(str(p))
            h, w = image.shape[:2]
            boxes, scores = run(image)
            results.append(ImageResult(gt=load_ground_truth(test / "labels" / f"{p.stem}.txt", w, h), boxes=boxes, scores=scores))
        s, best = score(results, exp.iou, exp.conf), best_f1_threshold(results, exp.iou)
        report[model_id] = {"all": s.as_dict(), "best_f1": best.as_dict(),
                            "by_group": breakdown([p.name for p in images], results, exp.iou, exp.conf)}
        print(f"{model_id}: P {s.precision:.3f} R {s.recall:.3f} AP50 {s.ap50:.3f} | best F1 at {best.conf:.2f}: P {best.precision:.3f} R {best.recall:.3f}")
        for g, v in report[model_id]["by_group"].items():
            print(f"   {g:28} {v['tp']}/{v['gt']} fp {v['fp']} P {v['precision']} R {v['recall']} AP {v['ap50']}")
    if args.out:
        import pickle

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Raw boxes and scores per model, to try ways of combining models without rerunning them.
        with args.out.with_suffix(".pkl").open("wb") as f:
            pickle.dump({"images": [p.name for p in images], "results": raw}, f)


if __name__ == "__main__":
    main()

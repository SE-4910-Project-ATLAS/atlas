"""Build an image-classification set of vehicle crops: ambulance or other vehicle.

    python -m atlas_training.crops configs/ambulance_v5.yaml --out datasets/crops_v5

Two-stage detection (Fan et al. 2022; Akdag et al., VEHITS 2022): a generic COCO detector
finds the vehicles, and a classifier looks at each vehicle crop, enlarged, to decide
whether it is an ambulance. On the Turkish test videos the COCO detector boxes 129 of 148
ambulances (IoU >= 0.5) while the one-stage ambulance detector v3 finds 52, so the miss
is in recognising the vehicle, not in finding it.

For every image of the prepared dataset (train, val and test kept apart):
  ambulance  each labelled ambulance box
  other      COCO car/bus/truck boxes that overlap no ambulance box (IoU < 0.1), at most
             --neg-per-pos per ambulance crop of the split: all of those from our own
             Turkish frames (the white vans and minibuses that matter), the rest at random
Each crop is the box with --pad of its size added on every side, made square and resized
to --size. The folder layout is the one Ultralytics classification training reads.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import config as cfg
from .evaluate import load_ground_truth
from .metrics import box_iou
from .prepare import IMAGE_SUFFIXES

VEHICLE_CLASSES = [2, 5, 7]  # COCO car, bus, truck


def square_crop(image: np.ndarray, box, pad: float) -> np.ndarray:
    """Square crop around box (x1 y1 x2 y2) with pad x size added per side, edge-padded."""
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    h, w = image.shape[:2]
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(1, int(round(side)))
    top, left = max(0, -b), max(0, -a)
    bottom, right = max(0, b + s - h), max(0, a + s - w)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return padded[b + top:b + top + s, a + left:a + left + s]


def negatives(vehicles: np.ndarray, scores: np.ndarray, ambulances: np.ndarray, max_iou: float = 0.1):
    """Indices of vehicle boxes that overlap no ambulance box."""
    if len(vehicles) == 0:
        return []
    if len(ambulances) == 0:
        return list(range(len(vehicles)))
    overlap = box_iou(vehicles, ambulances).max(axis=1)
    return [i for i in range(len(vehicles)) if overlap[i] < max_iou]


def sample_negatives(items: list, limit: int, rng: np.random.Generator, own_prefix: str = "own_cctv__") -> list:
    """All negatives from our own frames, then a random sample of the rest up to limit."""
    own = [x for x in items if x[0].name.startswith(own_prefix)]
    rest = [x for x in items if not x[0].name.startswith(own_prefix)]
    room = max(0, limit - len(own))
    if len(rest) > room:
        rest = [rest[i] for i in sorted(rng.choice(len(rest), size=room, replace=False))]
    return own + rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--det-conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min-side", type=int, default=12, help="smaller boxes are left out")
    parser.add_argument("--neg-per-pos", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    exp = cfg.load(args.config)
    detector = YOLO(str(args.detector))
    rng = np.random.default_rng(args.seed)
    rows = []
    for split in ("train", "val", "test"):
        folder = exp.dataset_dir / split
        positives, negs = [], []
        images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        # Oversampled copies (__r1, __r2) are the same image; crop each image once.
        images = [p for p in images if "__r" not in p.stem]
        for image_path in images:
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            gt = load_ground_truth(folder / "labels" / f"{image_path.stem}.txt", w, h)
            for k, box in enumerate(gt):
                if min(box[2] - box[0], box[3] - box[1]) >= args.min_side:
                    positives.append((image_path, box, k, 1.0))
            r = detector.predict(image, conf=args.det_conf, imgsz=args.imgsz, classes=VEHICLE_CLASSES,
                                 agnostic_nms=True, verbose=False)[0]
            boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            for i in negatives(boxes, scores, gt):
                b = boxes[i]
                if min(b[2] - b[0], b[3] - b[1]) >= args.min_side:
                    negs.append((image_path, b, i, float(scores[i])))
        negs = sample_negatives(negs, int(len(positives) * args.neg_per_pos), rng)
        for label, items in (("ambulance", positives), ("other", negs)):
            out = args.out / split / label
            out.mkdir(parents=True, exist_ok=True)
            for image_path, box, k, score in items:
                crop = square_crop(cv2.imread(str(image_path)), box, args.pad)
                crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
                name = f"{image_path.stem}_{label[0]}{k}.jpg"
                cv2.imwrite(str(out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append([split, label, name, image_path.name, *(round(float(v), 1) for v in box), round(score, 3)])
        print(f"{split}: {len(positives)} ambulance, {len(negs)} other")
    with (args.out / "crops.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "crop", "image", "x1", "y1", "x2", "y2", "score"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Where are ambulances lost: by the vehicle detector or by the crop classifier?

    python -m atlas_training.diagnose datasets/raw/own_cctv-v2/test \
        --classifier runs/cls/crops_v5e/weights/best.pt --out runs/eval/diagnose_dev.csv --device cpu

For every labelled ambulance box: its width, whether the frame is dark (night), whether
the COCO detector boxes it as a car/bus/truck (IoU >= --iou) at each --imgsz, and the
classifier's p(ambulance) on the detector's crop and on the labelled box's own crop. A
low p on the labelled crop means the classifier is the problem; a missing detector box
with a high p on the labelled crop means the detector is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .evaluate import class_index
from .metrics import box_iou

BUCKETS = ((0, 40), (40, 64), (64, 100), (100, 200), (200, 10**6))
NIGHT_V = 70  # mean HSV value of the frame below which it counts as night


def labelled_boxes(label: Path, w: int, h: int) -> np.ndarray:
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
    out = [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
           for _, x, y, bw, bh in ([float(v) for v in r[:5]] for r in rows)]
    return np.array(out, dtype=float).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path, help="folder with images/ and labels/ (YOLO format)")
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280])
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    detector = YOLO(str(args.detector))
    classifier = YOLO(str(args.classifier))
    amb = class_index(classifier.names, "ambulance")

    def p_of(crops: list[np.ndarray]) -> list[float]:
        if not crops:
            return []
        resized = [cv2.resize(c, (args.size, args.size)) for c in crops]
        return [r.probs.data[amb].item() for r in
                classifier.predict(resized, imgsz=args.size, device=args.device, verbose=False)]

    rows = []
    for label in sorted((args.split / "labels").glob("*.txt")):
        image_path = next((args.split / "images").glob(label.stem + ".*"), None)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        gt = labelled_boxes(label, w, h)
        if not len(gt):
            continue
        night = float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].mean()) < NIGHT_V
        p_gt = p_of([square_crop(image, b, args.pad) for b in gt])
        found: dict[int, list] = {}
        for size in args.imgsz:
            r = detector.predict(image, conf=args.conf, imgsz=size, classes=VEHICLE_CLASSES, agnostic_nms=True,
                                 device=args.device, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            overlap = box_iou(gt, boxes)
            best = [int(o.argmax()) if len(boxes) and o.max() >= args.iou else -1 for o in overlap]
            p_det = p_of([square_crop(image, boxes[j], args.pad) for j in best if j >= 0])
            it = iter(p_det)
            found[size] = [next(it) if j >= 0 else None for j in best]
        for k, b in enumerate(gt):
            row = {"frame": label.stem, "clip": label.stem.rsplit("_f", 1)[0], "width": round(float(b[2] - b[0]), 1),
                   "night": night, "p_gt": round(p_gt[k], 4)}
            for size in args.imgsz:
                row[f"det_{size}"] = found[size][k] is not None
                row[f"p_det_{size}"] = None if found[size][k] is None else round(found[size][k], 4)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summary(name: str, subset: list[dict]) -> None:
        if not subset:
            return
        parts = [f"{name:<14} n={len(subset):>3}", f"p_gt>=.5 {sum(r['p_gt'] >= 0.5 for r in subset):>3}"]
        for size in args.imgsz:
            det = [r for r in subset if r[f"det_{size}"]]
            ok = sum(r[f"p_det_{size}"] >= 0.5 for r in det)
            parts.append(f"@{size}: boxed {len(det):>3}, boxed and p>=.5 {ok:>3}")
        print(" | ".join(parts))

    for lo, hi in BUCKETS:
        summary(f"width {lo}-{hi if hi < 10**6 else ''}", [r for r in rows if lo <= r["width"] < hi])
    summary("night", [r for r in rows if r["night"]])
    summary("day", [r for r in rows if not r["night"]])
    summary("all", rows)


if __name__ == "__main__":
    main()

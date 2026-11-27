"""Take frames from video clips for labelling, with the model's boxes as pre-labels.

    python -m atlas_training.frames D:/code/atlas-data/clips/cctv/*.mp4 \
        --out D:/code/atlas-data/labeling/cctv_v1 --weights runs/train/ambulance_v2/weights/best.pt

Frames are read every --every seconds. A frame is kept only when it differs from the last
kept frame of the same clip (grey-thumbnail correlation below --similarity), so a parked
camera does not give hundreds of copies; if more remain than --max-per-video, evenly
spaced ones are kept. The output folder gets:

    images/<clip>_f<frame>.jpg
    labels/<clip>_f<frame>.txt   YOLO boxes "class cx cy w h confidence" from --weights
    frames.csv                   clip, frame, time and number of pre-labelled boxes

The pre-labels are suggestions only: every frame is checked and corrected by hand before
it is used (see atlas_training.labelstudio). Clips and frames are never committed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

THUMB = 32


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Grey 32x32 thumbnail, normalised for correlation."""
    import cv2

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    v = cv2.resize(grey, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def keep_distinct(thumbs: list[np.ndarray], similarity: float) -> list[int]:
    """Indices of thumbnails that differ from the last kept one (correlation < similarity)."""
    kept: list[int] = []
    for i, t in enumerate(thumbs):
        if not kept or float(t @ thumbs[kept[-1]]) < similarity:
            kept.append(i)
    return kept


def pick_evenly(items: list, n: int) -> list:
    """At most n items, evenly spaced, first and last included."""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def candidate_frames(video: Path, every: float, similarity: float, max_frames: int) -> tuple[list[int], float]:
    """Frame numbers (0-based) to keep from one video, and its frame rate."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(every * fps))
    numbers, thumbs = [], []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                numbers.append(index)
                thumbs.append(thumbnail(frame))
        index += 1
    capture.release()
    distinct = [numbers[i] for i in keep_distinct(thumbs, similarity)]
    return pick_evenly(distinct, max_frames), fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output folder, outside the repository")
    parser.add_argument("--every", type=float, default=0.5, help="seconds between frames read")
    parser.add_argument("--similarity", type=float, default=0.97, help="drop frames this similar to the last kept")
    parser.add_argument("--max-per-video", type=int, default=60)
    parser.add_argument("--weights", type=Path, help="model for pre-labels (its ambulance class)")
    parser.add_argument("--conf", type=float, default=0.15, help="lowest confidence kept as a pre-label")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    import cv2

    model = class_id = None
    if args.weights:
        from ultralytics import YOLO

        from .evaluate import class_index

        model = YOLO(str(args.weights))
        class_id = class_index(model.names, "ambulance")

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "clip", "frame", "time_s", "prelabels"])
        for video in args.videos:
            numbers, fps = candidate_frames(video, args.every, args.similarity, args.max_per_video)
            capture = cv2.VideoCapture(str(video))
            boxes_total = 0
            for number in numbers:
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok, frame = capture.read()
                if not ok:
                    continue
                name = f"{video.stem}_f{number:06d}"
                cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = []
                if model is not None:
                    r = model.predict(frame, conf=args.conf, classes=[class_id], imgsz=args.imgsz, verbose=False)[0]
                    for (cx, cy, w, h), conf in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist(), strict=True):
                        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.3f}")
                (args.out / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                boxes_total += len(lines)
                writer.writerow([f"{name}.jpg", video.stem, number, f"{number / fps:.2f}", len(lines)])
            capture.release()
            print(f"{video.name}: {len(numbers)} frames, {boxes_total} pre-labelled boxes")


if __name__ == "__main__":
    main()

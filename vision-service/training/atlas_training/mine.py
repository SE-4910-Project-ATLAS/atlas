"""Find frames worth labelling in videos from where the model flickers (hard examples).

    python -m atlas_training.mine D:/code/atlas-data/clips/tr/*.mp4 \
        --out D:/code/atlas-data/labeling/mined_v1 --weights runs/train/ambulance_v4/weights/best.pt

Following Jin et al., "Unsupervised Hard Example Mining from Videos for Improved Object
Detection" (ECCV 2018): an object that is really there is detected in consecutive frames,
so the model's mistakes show up as flicker.

  gap      the ambulance is detected before and after a frame but not in it: a likely
           missed ambulance (hard positive). The box is interpolated from its neighbours.
  flicker  a confident detection with no detection near it in the frames around it: a
           likely false alarm (hard negative), e.g. a white van for a few frames.

In the paper hard positives gave the largest gain and hard negatives alone could lower
recall, and about 75-90% of the mined examples were right, so every frame here is checked
by hand in Label Studio before it is used. The output folder has the same layout as
atlas_training.frames (images/, labels/ with the suggested boxes, frames.csv with the kind).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import box_iou


@dataclass
class Detections:
    """Ambulance boxes (x1 y1 x2 y2, pixels) and confidences of one frame."""

    boxes: np.ndarray
    scores: np.ndarray


@dataclass
class Example:
    index: int          # position in the list of read frames
    kind: str           # "gap" or "flicker"
    box: np.ndarray     # suggested ambulance box for a gap, the false box for a flicker


def _overlaps(box: np.ndarray, frame: Detections, iou: float) -> bool:
    return len(frame.boxes) > 0 and float(box_iou(box[None], frame.boxes).max()) >= iou


def find_flickers(frames: list[Detections], window: int, conf: float, iou: float) -> list[Example]:
    """Confident detections with no detection overlapping them in the `window` frames on
    either side. Frames near the start and end of the video are skipped, as they have no
    full window."""
    found = []
    for i in range(window, len(frames) - window):
        for box, s in zip(frames[i].boxes, frames[i].scores, strict=True):
            if s < conf:
                continue
            others = frames[i - window:i] + frames[i + 1:i + 1 + window]
            if not any(_overlaps(box, f, iou) for f in others):
                found.append(Example(i, "flicker", box))
    return found


def find_gaps(frames: list[Detections], window: int, conf: float, iou: float) -> list[Example]:
    """Frames with no detection where a confident detection before (within `window`) and a
    matching one after it make an ambulance very likely; the box is interpolated."""
    found = []
    for i in range(1, len(frames) - 1):
        for back in range(1, window + 1):
            j = i - back
            if j < 0:
                break
            before = frames[j]
            hit = None
            for box_b, s_b in zip(before.boxes, before.scores, strict=True):
                if s_b < conf:
                    continue
                for ahead in range(1, window + 1):
                    k = i + ahead
                    if k >= len(frames):
                        break
                    after = frames[k]
                    for box_a, s_a in zip(after.boxes, after.scores, strict=True):
                        if s_a >= conf and float(box_iou(box_b[None], box_a[None])[0, 0]) >= iou:
                            t = back / (back + ahead)
                            hit = box_b + t * (box_a - box_b)
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
            if hit is not None:
                if not _overlaps(hit, frames[i], iou):
                    found.append(Example(i, "gap", hit))
                break
    return found


def thin_out(examples: list[Example], min_distance: int, limit: int) -> list[Example]:
    """At most one example of each kind per `min_distance` frames, at most `limit` in all,
    so a long miss does not fill the set with near copies."""
    kept: list[Example] = []
    last: dict[str, int] = {}
    for e in sorted(examples, key=lambda e: e.index):
        if e.index - last.get(e.kind, -min_distance) >= min_distance:
            kept.append(e)
            last[e.kind] = e.index
    if len(kept) > limit:
        step = len(kept) / limit
        kept = [kept[int(i * step)] for i in range(limit)]
    return kept


def detect(video: Path, model, class_id: int, stride: int, imgsz: int, low_conf: float):
    """Run the model on every `stride`-th frame; returns frame numbers and Detections."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    numbers, frames = [], []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, image = capture.retrieve()
            if ok:
                r = model.predict(image, conf=low_conf, classes=[class_id], imgsz=imgsz, verbose=False)[0]
                numbers.append(index)
                frames.append(Detections(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        index += 1
    capture.release()
    return numbers, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output folder, outside the repository")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2, help="read every n-th frame")
    parser.add_argument("--window", type=int, default=5, help="frames on each side that are compared")
    parser.add_argument("--low-conf", type=float, default=0.25, help="detections below this count as none")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence of a detection that is trusted")
    parser.add_argument("--iou", type=float, default=0.2, help="overlap that makes two boxes the same object")
    parser.add_argument("--min-distance", type=int, default=12, help="read frames between kept examples")
    parser.add_argument("--max-per-video", type=int, default=40)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from .evaluate import class_index

    model = YOLO(str(args.weights))
    class_id = class_index(model.names, "ambulance")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "clip", "frame", "time_s", "kind", "prelabels"])
        for video in args.videos:
            numbers, frames = detect(video, model, class_id, args.stride, args.imgsz, args.low_conf)
            examples = (find_gaps(frames, args.window, args.conf, args.iou)
                        + find_flickers(frames, args.window, args.conf, args.iou))
            examples = thin_out(examples, args.min_distance, args.max_per_video)
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            for e in examples:
                number = numbers[e.index]
                capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok, image = capture.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                name = f"{video.stem}_f{number:06d}"
                cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Every detection of the frame is a suggestion, plus the interpolated box of
                # a gap (marked with confidence 0.0 so it stands out in Label Studio).
                det = frames[e.index]
                boxes = [(b, float(s)) for b, s in zip(det.boxes, det.scores, strict=True)]
                if e.kind == "gap":
                    boxes.append((e.box, 0.0))
                lines = []
                for (x1, y1, x2, y2), s in boxes:
                    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                    w, h = (x2 - x1) / width, (y2 - y1) / height
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.3f}")
                (args.out / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                writer.writerow([f"{name}.jpg", video.stem, number, f"{number / fps:.2f}", e.kind, len(lines)])
            capture.release()
            kinds = [e.kind for e in examples]
            print(f"{video.name}: {len(frames)} frames read, {kinds.count('gap')} gaps, "
                  f"{kinds.count('flicker')} flickers kept")


if __name__ == "__main__":
    main()

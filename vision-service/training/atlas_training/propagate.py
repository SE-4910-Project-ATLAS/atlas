"""More ambulance crops from the training videos by following labelled ambulances along their tracks.

    python -m atlas_training.propagate datasets/raw/own_cctv-v2/train/labels --out datasets/crops_prop

Our Turkish frames are labelled about once per second. Between two labelled frames the
same ambulance turns, comes closer or is seen from the front, which the classifier has
seen too little (see RESEARCH.md). Vehicles are tracked with ByteTrack; a track that
overlaps a labelled ambulance box (IoU >= --iou) is taken as that ambulance, and its
crops within --window seconds of such a labelled frame are kept as ambulance crops.
Interpolating sparse keyframe labels along tracks is the usual way to label video (VATIC;
Liu et al., ICCV 2023). Only training videos are used: test clips must not be listed.
Crops are written every --every-n read frames to limit near copies.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .metrics import box_iou

FRAME = re.compile(r"^(?P<clip>.+)_f(?P<frame>\d+)$")


def labelled_ambulances(labels_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    """clip -> {frame number: ambulance boxes as normalised cx cy w h}."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for f in labels_dir.glob("*.txt"):
        m = FRAME.match(f.stem)
        if not m:
            continue
        boxes = [list(map(float, line.split()[1:5])) for line in f.read_text().splitlines()
                 if line.strip() and line.split()[0] == "0"]
        out.setdefault(m["clip"], {})[int(m["frame"])] = np.asarray(boxes, float).reshape(-1, 4)
    return out


def to_xyxy(cxcywh: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = cxcywh.T
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)


def ambulance_tracks(frames: list[tuple[int, dict[int, np.ndarray]]], labels: dict[int, np.ndarray],
                     size: tuple[int, int], iou: float) -> dict[int, list[int]]:
    """track id -> labelled frame numbers at which the track matches a labelled ambulance."""
    w, h = size
    hits: dict[int, list[int]] = {}
    for number, tracks in frames:
        if number not in labels or not len(labels[number]) or not tracks:
            continue
        gt = to_xyxy(labels[number], w, h)
        ids = list(tracks)
        overlap = box_iou(np.stack([tracks[i] for i in ids]), gt)
        for k, track_id in enumerate(ids):
            if overlap[k].max() >= iou:
                hits.setdefault(track_id, []).append(number)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="labels folder of the training split")
    parser.add_argument("--clips", default="D:/code/atlas-data/clips/*/{clip}.mp4", help="glob pattern for a clip's video")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--tracker", type=Path, default=cfg.TRAINING_DIR / "configs" / "tracker_ambulance.yaml")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--window", type=float, default=1.5, help="seconds around a matched labelled frame")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--every-n", type=int, default=3, help="keep every n-th read frame of a track")
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min-side", type=int, default=24)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    labels = labelled_ambulances(args.labels)
    (args.out / "ambulance").mkdir(parents=True, exist_ok=True)
    total = 0
    for clip, clip_labels in sorted(labels.items()):
        if not any(len(b) for b in clip_labels.values()):
            continue
        found = glob.glob(args.clips.format(clip=clip))
        if not found:
            print(f"{clip}: video not found, skipped")
            continue
        detector = YOLO(str(args.detector))
        capture = cv2.VideoCapture(found[0])
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames, index = [], 0
        while True:
            ok = capture.grab()
            if not ok:
                break
            # Labelled frames are always read, so they can be matched exactly.
            if index % args.stride == 0 or index in clip_labels:
                ok, image = capture.retrieve()
                if not ok:
                    break
                r = detector.track(image, persist=True, tracker=str(args.tracker), conf=0.1, imgsz=960,
                                   classes=VEHICLE_CLASSES, agnostic_nms=True, verbose=False)[0]
                tracks = {}
                if r.boxes is not None and r.boxes.id is not None:
                    tracks = dict(zip(r.boxes.id.int().cpu().tolist(), r.boxes.xyxy.cpu().numpy(), strict=True))
                frames.append((index, tracks))
            index += 1
        hits = ambulance_tracks(frames, clip_labels, size, args.iou)
        keep: dict[int, dict[int, np.ndarray]] = {}
        count: dict[int, int] = {}
        for number, tracks in frames:
            for track_id, box in tracks.items():
                near = any(abs(number - m) <= args.window * fps for m in hits.get(track_id, []))
                if not near or min(box[2] - box[0], box[3] - box[1]) < args.min_side:
                    continue
                count[track_id] = count.get(track_id, 0) + 1
                if count[track_id] % args.every_n == 0 or number in clip_labels:
                    keep.setdefault(number, {})[track_id] = box
        n = 0
        for number in sorted(keep):
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, image = capture.read()
            if not ok:
                continue
            for track_id, box in keep[number].items():
                crop = cv2.resize(square_crop(image, box, args.pad), (args.size, args.size))
                cv2.imwrite(str(args.out / "ambulance" / f"prop_{clip}_f{number:06d}_t{track_id}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
        capture.release()
        total += n
        print(f"{clip}: {len(hits)} ambulance tracks, {n} crops", flush=True)
    print(f"total {total} propagated ambulance crops")


if __name__ == "__main__":
    main()

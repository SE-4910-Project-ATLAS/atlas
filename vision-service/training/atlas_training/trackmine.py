"""Mine classifier crops along vehicle tracks, for checking by eye (two-stage system).

    python -m atlas_training.trackmine D:/code/atlas-data/clips/tr3/*.mp4 \
        --classifier runs/cls/crops_v5e/weights/best.pt --out D:/code/atlas-data/labeling/trackmine_v1

Vehicles are tracked as in atlas_training.tracks. Every track whose classifier score
reaches --candidate-p in some frame is a candidate; all of its crops (224 px, the input the
classifier sees) are saved as <out>/<video>/<track>/<time>_<p>.jpg, and one contact sheet
per video (<out>/sheets/<video>.jpg) shows a strip of each candidate track with its scores.

A track is then marked by hand in <out>/verdicts.csv (video,track,verdict) as
ambulance, other or skip (mixed, unclear, a track that jumped to another vehicle):

- ambulance tracks give hard positives: the frames where the classifier scored low are
  views it does not know yet (head-on, at night, door open; Jin et al., ECCV 2018: missed
  frames of a tracked object are the most useful new examples);
- other tracks give hard negatives: white vans and minibuses it took for an ambulance.

Propagating labels without this check (classifier v5f) taught van false alarms, so no
crop is used unless its whole track was looked at.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .evaluate import class_index


def mine_video(video: Path, detector, classifier, amb: int, args) -> dict[int, list[tuple[float, float, np.ndarray]]]:
    """{track id: [(time, p, crop)]} for candidate tracks of one video."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / args.target_fps))
    tracks: dict[int, list] = {}
    best: dict[int, float] = {}
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, image = capture.retrieve()
            if not ok:
                break
            r = detector.track(image, persist=True, tracker=str(args.tracker), conf=args.det_conf, imgsz=args.imgsz,
                               classes=VEHICLE_CLASSES, agnostic_nms=True, verbose=False)[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().tolist()
                keep = [k for k, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= args.min_side]
                crops = [cv2.resize(square_crop(image, boxes[k], args.pad), (args.size, args.size)) for k in keep]
                if crops:
                    ps = [c.probs.data[amb].item() for c in classifier.predict(crops, imgsz=args.size, verbose=False)]
                    for k, crop, p in zip(keep, crops, ps, strict=True):
                        tid = ids[k]
                        tracks.setdefault(tid, []).append((index / fps, p, crop))
                        best[tid] = max(best.get(tid, 0.0), p)
        index += 1
    capture.release()
    return {t: v for t, v in tracks.items() if best[t] >= args.candidate_p and len(v) >= args.min_frames}


def contact_sheet(tracks: dict[int, list], path: Path, per_track: int = 10) -> None:
    import cv2

    rows = []
    for tid, items in sorted(tracks.items()):
        pick = [items[i] for i in np.linspace(0, len(items) - 1, min(per_track, len(items))).astype(int)]
        tiles = []
        for t, p, crop in pick:
            tile = cv2.resize(crop, (112, 112))
            cv2.putText(tile, f"{p:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(tile)
        tiles += [np.zeros((112, 112, 3), np.uint8)] * (per_track - len(tiles))
        mean = float(np.mean([p for _, p, _ in items]))
        label = np.zeros((112, 150, 3), np.uint8)
        cv2.putText(label, f"track {tid}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(label, f"n={len(items)} mean={mean:.2f}", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(label, f"t={items[0][0]:.0f}-{items[-1][0]:.0f}s", (4, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        rows.append(np.hstack([label] + tiles))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--tracker", type=Path, default=cfg.TRAINING_DIR / "configs" / "tracker_ambulance.yaml")
    parser.add_argument("--det-conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--min-side", type=int, default=20, help="smaller boxes are left out")
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--candidate-p", type=float, default=0.3)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True, help="output folder, outside the repository")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    classifier = YOLO(str(args.classifier))
    amb = class_index(classifier.names, "ambulance")
    summary = []
    for video in args.videos:
        detector = YOLO(str(args.detector))  # fresh tracker state per video
        tracks = mine_video(video, detector, classifier, amb, args)
        for tid, items in tracks.items():
            folder = args.out / video.stem / str(tid)
            folder.mkdir(parents=True, exist_ok=True)
            for t, p, crop in items:
                cv2.imwrite(str(folder / f"{t:08.2f}_{p:.3f}.jpg"), crop)
            ps = [p for _, p, _ in items]
            summary.append({"video": video.stem, "track": tid, "frames": len(items), "mean_p": round(float(np.mean(ps)), 3),
                            "max_p": round(max(ps), 3), "start_s": round(items[0][0], 1), "end_s": round(items[-1][0], 1)})
        contact_sheet(tracks, args.out / "sheets" / f"{video.stem}.jpg")
        print(f"{video.stem}: {len(tracks)} candidate tracks", flush=True)
    with (args.out / "tracks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "track", "frames", "mean_p", "max_p", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()

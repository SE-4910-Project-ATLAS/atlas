"""Run a model on traffic camera clips and record what it detects (S4PT-52, step 1).

    python -m atlas_training.clips weights/smarttrafficai.pt clip1.mp4 clip2.mp4

For each clip, runs/clips/<model>/ gets an annotated video and a CSV with one row per
detection (frame, time, class, confidence, box). A short summary per clip and class is
printed: in how many frames each class was seen and its highest confidence. Missed
vehicles are not visible in the numbers; note them while watching the annotated video.
Clips are recorded footage (S4PT-51) and are never committed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from . import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    out = cfg.RUNS_DIR / "clips" / args.weights.stem
    out.mkdir(parents=True, exist_ok=True)

    for clip in args.clips:
        capture = cv2.VideoCapture(str(clip))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        capture.release()
        seen = defaultdict(set)
        best = defaultdict(float)
        csv_path = out / f"{clip.stem}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time_s", "class", "confidence", "x1", "y1", "x2", "y2"])
            results = model.predict(str(clip), conf=args.conf, imgsz=args.imgsz, stream=True,
                                    save=True, project=str(out), name=clip.stem, exist_ok=True,
                                    verbose=False)
            frames = 0
            for frames, r in enumerate(results, start=1):
                time_s = f"{(frames - 1) / fps:.2f}" if fps else ""
                for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(),
                                          strict=True):
                    name = model.names[int(cls)]
                    seen[name].add(frames)
                    best[name] = max(best[name], conf)
                    writer.writerow([frames, time_s, name, f"{conf:.3f}", *(f"{v:.1f}" for v in box)])
        print(f"\n{clip.name}: {frames} frames, detections in {csv_path}")
        for name in sorted(seen):
            print(f"  {name}: seen in {len(seen[name])} frames, highest confidence {best[name]:.2f}")
        if not seen:
            print("  nothing detected")


if __name__ == "__main__":
    main()

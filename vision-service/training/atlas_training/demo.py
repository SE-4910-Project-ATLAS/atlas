"""Demo video: the final two-stage system drawn on a clip, for watching and presenting.

    python -m atlas_training.demo D:/code/atlas-data/clips/test2/X.mp4 \
        --classifier runs/cls/crops_v5g/weights/best.pt --out D:/code/atlas-data/demo/X_demo.mp4

Every read frame (--fps per second) goes through the same chain as atlas_training.tracks:
COCO vehicle boxes -> ByteTrack -> 224 px square crop -> classifier p(ambulance). Each
vehicle is drawn with its track number, its frame score p and its running track mean; a
track that meets the chosen rule (mean >= --threshold over >= --min-frames frames) is
drawn red and an AMBULANS banner is shown while it is in view. The place memory of the
final rule (one alarm per place in 30 min) only matters on long fixed-camera recordings
and is not drawn here. The output stays local: clips from YouTube are never published.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .evaluate import class_index
from .tracks import TrackVotes

GREY, ORANGE, RED, WHITE = (170, 170, 170), (0, 165, 255), (0, 0, 230), (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--tracker", type=Path, default=cfg.TRAINING_DIR / "configs" / "tracker_ambulance.yaml")
    parser.add_argument("--det-conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--fps", type=float, default=12.5, help="frames read and written per second")
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--show-below", type=float, default=0.1, help="vehicles under this mean are drawn thin, no label")
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, help="seconds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    detector = YOLO(str(args.detector))
    classifier = YOLO(str(args.classifier))
    amb = class_index(classifier.names, "ambulance")

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / args.fps))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    index = round(args.start * fps)
    votes = TrackVotes()
    alarmed: set[int] = set()
    ever: set[int] = set()
    scale = max(0.8, width / 800)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok = capture.grab()
        if not ok or (args.end is not None and index / fps > args.end):
            break
        if index % stride:
            index += 1
            continue
        ok, image = capture.retrieve()
        if not ok:
            break
        r = detector.track(image, persist=True, tracker=str(args.tracker), conf=args.det_conf, imgsz=args.imgsz,
                           classes=VEHICLE_CLASSES, agnostic_nms=True, device=args.device, verbose=False)[0]
        shown = []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            crops = [cv2.resize(square_crop(image, b, args.pad), (args.size, args.size)) for b in boxes]
            ps = [c.probs.data[amb].item() for c in
                  classifier.predict(crops, imgsz=args.size, device=args.device, verbose=False)]
            for box, tid, p in zip(boxes, ids, ps, strict=True):
                votes.add(tid, p)
                # As in the rule itself, a track is an ambulance while its mean stays high;
                # a track whose mean falls back (e.g. an identity switch) is no longer drawn red.
                if votes.is_ambulance(tid, args.min_frames, args.threshold):
                    alarmed.add(tid)
                    ever.add(tid)
                else:
                    alarmed.discard(tid)
                shown.append((box, tid, p, votes.total[tid] / votes.count[tid]))
        canvas = image.copy()
        alarm_now, alarm_top = False, height
        for box, tid, p, mean in shown:
            x1, y1, x2, y2 = (int(v) for v in box)
            if tid in alarmed:
                colour, thick, alarm_now = RED, max(3, int(3 * scale)), True
                alarm_top = min(alarm_top, y1)
            elif mean >= 0.3:
                colour, thick = ORANGE, max(2, int(2 * scale))
            else:
                colour, thick = GREY, 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thick)
            if mean < args.show_below and tid not in alarmed:
                continue
            text = f"#{tid} p={p:.2f} ort={mean:.2f} n={votes.count[tid]}"
            (tw, th), _ = cv2.getTextSize(text, font, 0.5 * scale, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(canvas, text, (x1 + 3, y1 - 5), font, 0.5 * scale, WHITE, 1, cv2.LINE_AA)
        if alarm_now:
            # The banner goes to the bottom when it would cover the ambulance at the top.
            bh = int(60 * scale)
            y0 = 0 if alarm_top > bh * 1.3 else height - bh - int(30 * scale)
            cv2.rectangle(canvas, (0, y0), (width, y0 + bh), RED, -1)
            cv2.putText(canvas, "AMBULANS", (int(20 * scale), y0 + int(45 * scale)), font, 1.4 * scale, WHITE,
                        max(2, int(3 * scale)), cv2.LINE_AA)
        footer = f"t={index / fps:6.1f}s  araclar={len(shown)}  kural: ort>={args.threshold} & n>={args.min_frames}"
        cv2.putText(canvas, footer, (10, height - 12), font, 0.5 * scale, WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, footer, (10, height - 12), font, 0.5 * scale, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(canvas)
        index += 1
    capture.release()
    writer.release()
    print(f"{args.out}: {len(ever)} alarmed tracks", flush=True)


if __name__ == "__main__":
    main()

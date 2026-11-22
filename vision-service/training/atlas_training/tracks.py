"""Track-level ambulance decision and clip-level scoring (NFR-04).

    python -m atlas_training.tracks D:/code/atlas-data/clips/cctv/QiYlYgw7P18.mp4 ... \
        --labels datasets/raw/own_cctv-v2/test/labels --classifier runs/cls/crops_v5b/weights/best.pt

Vehicles (COCO car, bus, truck) are tracked with ByteTrack; each crop gets p(ambulance)
from the classifier. A track is an ambulance when it has at least --min-frames frames and
the **mean** of its probabilities is at least --threshold (sequence rescoring with the
average, Seq-NMS, Han et al. 2016; scores pooled along a track, Kang et al. 2018). The
alarm of a frame is on while such a track is visible. With --one-stage, a box of that
detector (e.g. v3) overlapping a track lifts that frame's probability to its score
(the frame-level fusion of atlas_training.twostage).

Clip-level scoring uses atlas_training.events: labelled ambulance passages with an alarm
inside them (+/- --margin s), time to the first alarm, and alarm spans outside any passage
(false alarms; per hour on clips without ambulances).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as cfg
from .crops import VEHICLE_CLASSES, square_crop
from .lights import lit_fractions
from .events import Event, events_from_labels, labelled_times, score_video, shot_cuts, split_at_cuts
from .evaluate import class_index
from .metrics import box_iou


class TrackVotes:
    """Per-track running mean of p(ambulance)."""

    def __init__(self):
        self.total: dict[int, float] = {}
        self.count: dict[int, int] = {}

    def add(self, track_id: int, p: float) -> None:
        self.total[track_id] = self.total.get(track_id, 0.0) + p
        self.count[track_id] = self.count.get(track_id, 0) + 1

    def is_ambulance(self, track_id: int, min_frames: int, threshold: float) -> bool:
        n = self.count.get(track_id, 0)
        return n >= min_frames and self.total[track_id] / n >= threshold


def run_video(video: Path, detector, classifier, amb: int, one_stage, args) -> tuple[list[float], list[dict]]:
    """Per read frame: time and {track id: p(ambulance)} for the vehicles tracked in it."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    # A fixed processing rate whatever the video's frame rate, so that --min-frames means
    # the same time on every clip (NFR-02: >= 5 FPS per stream).
    stride = max(1, round(fps / args.target_fps)) if args.target_fps else args.stride
    # Candidate mode (NFR-02: verification at the stream's full rate on cameras with an
    # emergency-vehicle candidate): while any track scores >= --candidate-p, read at
    # --candidate-fps for --candidate-hold seconds.
    fast = max(1, round(fps / args.candidate_fps)) if args.candidate_fps else stride
    fast_until = -1.0
    times, frames, raw_frames, geo_frames, light_frames = [], [], [], [], []
    index, last_read = 0, -10**9
    while True:
        ok = capture.grab()
        if not ok:
            break
        step = fast if index / fps <= fast_until else stride
        if index - last_read >= step:
            last_read = index
            ok, image = capture.retrieve()
            if not ok:
                break
            r = detector.track(image, persist=True, tracker=str(args.tracker), conf=args.det_conf,
                               imgsz=args.imgsz, classes=VEHICLE_CLASSES, agnostic_nms=True, verbose=False)[0]
            probs: dict[int, float] = {}
            raw: dict[int, float] = {}
            geo: dict[int, list[float]] = {}
            lit: dict[int, list[float]] = {}
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().tolist()
                crops = [cv2.resize(square_crop(image, b, args.pad), (args.size, args.size)) for b in boxes]
                ps = [c.probs.data[amb].item() for c in classifier.predict(crops, imgsz=args.size, verbose=False)]
                raw = dict(zip(ids, ps, strict=True))
                # Box centre and width (pixels), for motion rules applied after the run.
                geo = {i: [round(float((b[0] + b[2]) / 2), 1), round(float((b[1] + b[3]) / 2), 1),
                           round(float(b[2] - b[0]), 1)] for i, b in zip(ids, boxes, strict=True)}
                # Lit red and blue share of the box, for the flashing-light rule (lights.py).
                lit = {i: [round(v, 4) for v in lit_fractions(image, b)] for i, b in zip(ids, boxes, strict=True)}
                if args.tta:
                    # Test-time augmentation: average with the mirrored crop. The unflipped
                    # scores are kept too, so one run gives both results.
                    flipped = [c.probs.data[amb].item() for c in
                               classifier.predict([cv2.flip(x, 1) for x in crops], imgsz=args.size, verbose=False)]
                    ps = [(a + b) / 2 for a, b in zip(ps, flipped, strict=True)]
                if one_stage is not None:
                    model, idx, size = one_stage
                    o = model.predict(image, conf=0.25, classes=[idx], imgsz=size, verbose=False)[0]
                    ob, os_ = o.boxes.xyxy.cpu().numpy(), o.boxes.conf.cpu().numpy()
                    if len(ob):
                        overlap = box_iou(boxes, ob)
                        for k in range(len(ps)):
                            j = int(overlap[k].argmax())
                            if overlap[k, j] >= 0.5:
                                ps[k] = max(ps[k], float(os_[j]))
                probs = dict(zip(ids, ps, strict=True))
                if args.candidate_fps and max(ps, default=0.0) >= args.candidate_p:
                    fast_until = index / fps + args.candidate_hold
            times.append(index / fps)
            frames.append(probs)
            raw_frames.append(raw)
            geo_frames.append(geo)
            light_frames.append(lit)
        index += 1
    capture.release()
    return times, frames, raw_frames, geo_frames, light_frames


def alarm_sprt(frames: list[dict], upper: float, clip: float = 2.0, prior: float = 0.5) -> list[bool]:
    """Wald's sequential probability ratio test per track: the evidence of a frame is the
    log-odds of its classifier probability minus the log-odds of `prior`, clipped to
    +/- `clip` so that one over-confident frame cannot decide alone (frames of a track are
    correlated, so a plain sum would overcount them). A track becomes an ambulance when its
    summed evidence reaches `upper`; it stays one while visible (Koch et al., CVPR-W 2013:
    multiple-look fusion with the SPRT)."""
    import math

    base = math.log(prior / (1 - prior))
    total: dict[int, float] = {}
    decided: set[int] = set()
    alarm = []
    for probs in frames:
        for track_id, p in probs.items():
            p = min(max(p, 1e-6), 1 - 1e-6)
            total[track_id] = total.get(track_id, 0.0) + max(-clip, min(clip, math.log(p / (1 - p)) - base))
            if total[track_id] >= upper:
                decided.add(track_id)
        alarm.append(any(t in decided for t in probs))
    return alarm


def alarm_from_tracks_time(times: list[float], frames: list[dict], min_seconds: float, threshold: float,
                           min_frames: int = 3) -> list[bool]:
    """Like alarm_from_tracks, but a track must have been seen for at least `min_seconds`
    (and `min_frames` frames), so the rule means the same time whatever the read rate:
    with a variable rate, a frame count would be easier to reach while reading faster."""
    total: dict[int, float] = {}
    count: dict[int, int] = {}
    first: dict[int, float] = {}
    alarm = []
    for t, probs in zip(times, frames, strict=True):
        on = False
        for track_id, p in probs.items():
            total[track_id] = total.get(track_id, 0.0) + p
            count[track_id] = count.get(track_id, 0) + 1
            first.setdefault(track_id, t)
            n = count[track_id]
            if n >= min_frames and t - first[track_id] >= min_seconds and total[track_id] / n >= threshold:
                on = True
        alarm.append(on)
    return alarm


def moving_tracks(boxes: list[dict], min_move: float) -> set[int]:
    """Tracks whose box centre moved at least `min_move` box widths from where the track
    started, at any time. A parked vehicle under shop lighting can be lost and found again
    by the tracker many times; without this rule each new track is a new alarm, and a
    parked vehicle is not one drivers must give way to (FR-08 alerts use speed and
    direction)."""
    start: dict[int, tuple[float, float]] = {}
    moved: set[int] = set()
    for frame in boxes:
        for key, (cx, cy, w) in frame.items():
            track_id = int(key)
            x0, y0 = start.setdefault(track_id, (cx, cy))
            if ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5 >= min_move * max(w, 1.0):
                moved.add(track_id)
    return moved


def only_moving(frames: list[dict], boxes: list[dict], min_move: float) -> list[dict]:
    """Per-frame scores restricted to tracks that have moved by that frame."""
    start: dict[int, tuple[float, float]] = {}
    moved: set[int] = set()
    out = []
    for probs, frame in zip(frames, boxes, strict=True):
        for key, (cx, cy, w) in frame.items():
            track_id = int(key)
            x0, y0 = start.setdefault(track_id, (cx, cy))
            if ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5 >= min_move * max(w, 1.0):
                moved.add(track_id)
        out.append({t: p for t, p in probs.items() if t in moved})
    return out


def alarm_once_per_place(times: list[float], frames: list[dict], boxes: list[dict], min_frames: int,
                         threshold: float, radius: float = 0.5, window: float = 1800.0) -> list[bool]:
    """The mean rule, but a place alarms once: a track that meets the rule where an earlier
    alarm was raised (centre within `radius` box widths, similar width) in the last
    `window` seconds is held back while it stays there, and released once it moves away.

    A parked vehicle under shop light is lost and found again by the tracker many times;
    with the plain rule each new track is a new alarm. A moving ambulance that crosses an
    earlier alarm's place is only held back until it leaves it. Same idea as a memory of
    false-positive places on a fixed camera (US12430919; BEM, arXiv 2604.11714), but kept
    to places this run has alarmed at, with no training."""
    votes = TrackVotes()
    released: set[int] = set()
    places: list[list[float]] = []  # [cx, cy, w, last time seen]
    alarm = []
    for t, probs, frame in zip(times, frames, boxes, strict=True):
        for track_id, p in probs.items():
            votes.add(track_id, p)
        on = False
        for track_id in probs:
            if not votes.is_ambulance(track_id, min_frames, threshold):
                continue
            box = frame.get(track_id) or frame.get(str(track_id))
            if box is None:
                on = on or track_id in released
                continue
            cx, cy, w = box
            match = None
            for place in places:
                px, py, pw, seen = place
                if t - seen <= window and 0.67 <= w / max(pw, 1.0) <= 1.5 \
                        and ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 <= radius * max(pw, 1.0):
                    match = place
                    break
            if match is not None:
                match[3] = t
                if track_id not in released:
                    continue
            elif track_id not in released:
                released.add(track_id)
                places.append([cx, cy, w, t])
            on = on or track_id in released
        alarm.append(on)
    return alarm


def alarm_from_tracks(frames: list[dict], min_frames: int, threshold: float) -> list[bool]:
    votes, alarm = TrackVotes(), []
    for probs in frames:
        for track_id, p in probs.items():
            votes.add(track_id, p)
        alarm.append(any(votes.is_ambulance(t, min_frames, threshold) for t in probs))
    return alarm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--labels", type=Path, help="labelled frames; clips without labels count as ambulance-free")
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=cfg.WEIGHTS_DIR / "yolo11s.pt")
    parser.add_argument("--one-stage", metavar="WEIGHTS@IMGSZ", help="also use this detector's boxes, e.g. v3")
    parser.add_argument("--tracker", type=Path, default=cfg.TRAINING_DIR / "configs" / "tracker_ambulance.yaml")
    parser.add_argument("--det-conf", type=float, default=0.1, help="low, so ByteTrack gets its low-score boxes")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=2, help="read every n-th frame (ignored with --target-fps)")
    parser.add_argument("--target-fps", type=float, help="read about this many frames per second, e.g. 5 (NFR-02)")
    parser.add_argument("--tta", action="store_true", help="average the classifier score with the mirrored crop")
    parser.add_argument("--candidate-fps", type=float, help="rate while a candidate is seen, e.g. 12.5")
    parser.add_argument("--candidate-p", type=float, default=0.3, help="track score that makes a candidate")
    parser.add_argument("--candidate-hold", type=float, default=3.0, help="seconds the faster rate is kept")
    parser.add_argument("--min-frames", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True, help="JSON with per-frame track probabilities and scores")
    parser.add_argument("--rescore", type=Path, help="score the frames saved in this JSON instead of running the models")
    parser.add_argument("--intervals", type=Path, help="clip labels as time intervals (JSON, see test2/labels.json) instead of --labels")
    args = parser.parse_args()
    if args.rescore:
        runs = json.loads(args.rescore.read_text(encoding="utf-8"))["runs"]
        cuts = {v.stem: shot_cuts(v) for v in args.videos if v.stem in runs}
        write_scores(runs, cuts, args)
        return

    import cv2
    from ultralytics import YOLO

    classifier = YOLO(str(args.classifier))
    amb = class_index(classifier.names, "ambulance")
    one_stage = None
    if args.one_stage:
        path, _, size = args.one_stage.partition("@")
        model = YOLO(path)
        one_stage = (model, class_index(model.names, "ambulance"), int(size or 640))

    runs = {}
    for video in args.videos:
        detector = YOLO(str(args.detector))  # a fresh tracker state per video
        times, frames, raw_frames, geo_frames, light_frames = run_video(video, detector, classifier, amb, one_stage, args)
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        labelled = labelled_times(args.labels, video.stem, fps) if args.labels else ([], [])
        runs[video.stem] = {"times": times, "frames": [{str(k): round(v, 4) for k, v in f.items()} for f in frames],
                            "frames_no_tta": [{str(k): round(v, 4) for k, v in f.items()} for f in raw_frames],
                            "boxes": [{str(k): v for k, v in g.items()} for g in geo_frames],
                            "lights": [{str(k): v for k, v in g.items()} for g in light_frames],
                            "with_ambulance": labelled[0], "without": labelled[1]}
        if args.intervals:
            spec = json.loads(args.intervals.read_text(encoding="utf-8"))["videos"][video.stem]
            runs[video.stem]["intervals"] = spec
        print(f"{video.stem}: {len(frames)} frames read", flush=True)
    write_scores(runs, {v.stem: shot_cuts(v) for v in args.videos}, args)


def write_scores(runs: dict, cuts: dict, args) -> None:
    """Sweep min_frames and threshold. Two levels are reported: per passage (labelled
    ambulance frames of one shot) and per clip (a video with an ambulance counts as found
    when any of its passages is found; a video without one counts a false alarm when the
    alarm is ever on)."""
    sweep = []
    for min_frames in args.min_frames:
        for threshold in args.threshold:
            total = {"events": 0, "found": 0, "false_alarms": 0, "duration_s": 0.0, "delays": []}
            clips = {"positive": 0, "found": 0, "negative": 0, "false": 0}
            per_clip = {}
            for clip, r in runs.items():
                frames = [{int(k): v for k, v in f.items()} for f in r["frames"]]
                alarm = alarm_from_tracks(frames, min_frames, threshold)
                if "intervals" in r:
                    events = [Event(a, b) for a, b in r["intervals"]["positive"]]
                    ignored = r["intervals"]["ignore"]
                    alarm = [a and not any(lo <= t <= hi for lo, hi in ignored) for t, a in zip(r["times"], alarm, strict=True)]
                else:
                    events = events_from_labels(r["with_ambulance"], r["without"], args.merge)
                    events = split_at_cuts(events, cuts.get(clip, []))
                s = score_video(events, r["times"], alarm, args.margin)
                per_clip[clip] = s
                for k in ("events", "found", "false_alarms", "duration_s"):
                    total[k] += s[k]
                total["delays"] += s["delays_s"]
                if events:
                    clips["positive"] += 1
                    clips["found"] += s["found"] > 0
                else:
                    clips["negative"] += 1
                    clips["false"] += any(alarm)
            recall = total["found"] / total["events"] if total["events"] else 0.0
            precision = total["found"] / (total["found"] + total["false_alarms"]) if total["found"] + total["false_alarms"] else 0.0
            row = {"min_frames": min_frames, "threshold": threshold, "events": total["events"], "found": total["found"],
                   "false_alarms": total["false_alarms"], "clip_recall": round(recall, 3), "clip_precision": round(precision, 3),
                   "median_delay_s": round(float(np.median(total["delays"])), 2) if total["delays"] else None,
                   "clips": clips, "per_clip": per_clip}
            sweep.append(row)
            print(f"min_frames {min_frames} threshold {threshold}: passages {total['found']}/{total['events']} "
                  f"(recall {recall:.3f}), false alarm spans {total['false_alarms']}, precision {precision:.3f} | "
                  f"clips with ambulance {clips['found']}/{clips['positive']}, "
                  f"clips without ambulance with an alarm {clips['false']}/{clips['negative']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs, "sweep": sweep}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

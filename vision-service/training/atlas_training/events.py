"""Score the detector the way the system uses it: per ambulance passage, not per frame.

    python -m atlas_training.events D:/code/atlas-data/clips/tr/A_mp-v1pY-M.mp4 ... \
        --labels datasets/raw/own_cctv-v1/test/labels --weights runs/train/ambulance_v4/weights/best.pt

An ambulance is often hidden for a few frames (behind a bus, a tree, the frame edge), so
"found in every frame" is not the goal; the goal is an alarm soon after it appears, and no
alarm without one. Per-frame detections go through a smoother like the one of Fan et al.,
"Detection of Active Emergency Vehicles using Per-Frame CNNs and Output Smoothing" (2022):
over the last `buffer` read frames, the alarm is on when at least `min_frames` frames have a
detection and they are at least `ratio` of the frames read. In that paper a ratio of 0.3-0.5
cut false positives by about 10% for a 0.5-0.7% loss of recall, and 0.7 lost 20% of recall.

Ground truth comes from the hand-labelled frames of each video (<clip>_f<frame>.txt): an
event is a run of labelled frames with an ambulance, split where the gap between them is
longer than --merge seconds. Results: events found, time to the first alarm, and alarm time
outside any event (a false alarm), also per hour for footage without ambulances.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Event:
    start: float    # seconds
    end: float


def smooth(hits: list[bool], buffer: int, min_frames: int, ratio: float) -> list[bool]:
    """Alarm state after each read frame from per-frame 'ambulance detected' flags."""
    out, window = [], []
    for hit in hits:
        window.append(hit)
        if len(window) > buffer:
            window.pop(0)
        n = sum(window)
        out.append(n >= min_frames and n / len(window) >= ratio)
    return out


def events_from_labels(times_with_ambulance: list[float], times_without: list[float], merge: float) -> list[Event]:
    """Runs of labelled frames with an ambulance; a labelled frame without one, or a gap of
    more than `merge` seconds, ends a run."""
    marks = sorted([(t, True) for t in times_with_ambulance] + [(t, False) for t in times_without])
    events: list[Event] = []
    current: Event | None = None
    for t, has in marks:
        if has:
            if current is not None and t - current.end <= merge:
                current.end = t
            else:
                if current is not None:
                    events.append(current)
                current = Event(t, t)
        elif current is not None:
            events.append(current)
            current = None
    if current is not None:
        events.append(current)
    return events


def split_at_cuts(events: list[Event], cuts: list[float]) -> list[Event]:
    """Merge events of the same shot into one passage and never merge across a cut: news
    videos cut between camera views, and the sparse labels (about one per second) would
    otherwise turn one ambulance into several passages or join two shots into one."""
    out: list[Event] = []
    bounds = [float("-inf"), *sorted(cuts), float("inf")]
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        inside = [e for e in events if lo <= e.start < hi]
        if inside:
            out.append(Event(min(e.start for e in inside), max(e.end for e in inside)))
    return out


def shot_cuts(video, every: float = 0.2, threshold: float = 0.5) -> list[float]:
    """Times (s) of hard cuts: HSV-histogram correlation between frames `every` seconds
    apart below `threshold` (colour-histogram shot detection, Lienhart 1999)."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(every * fps))
    cuts, previous, index = [], None, 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if previous is not None and cv2.compareHist(previous, hist, cv2.HISTCMP_CORREL) < threshold:
                    cuts.append(index / fps)
                previous = hist
        index += 1
    capture.release()
    return cuts


def alarm_spans(times: list[float], alarm: list[bool]) -> list[Event]:
    """Runs of consecutive read frames with the alarm on."""
    spans: list[Event] = []
    previous_on = False
    for t, on in zip(times, alarm, strict=True):
        if on and previous_on:
            spans[-1].end = t
        elif on:
            spans.append(Event(t, t))
        previous_on = on
    return spans


def score_video(events: list[Event], times: list[float], alarm: list[bool], margin: float) -> dict:
    """Events found (alarm on between start and end + margin), seconds to the first alarm,
    and false-alarm spans: alarm spans that do not touch any event (+/- margin)."""
    found, delays = 0, []
    for e in events:
        on = [t for t, a in zip(times, alarm, strict=True) if a and e.start - margin <= t <= e.end + margin]
        if on:
            found += 1
            delays.append(max(0.0, on[0] - e.start))
    false_spans = [
        s for s in alarm_spans(times, alarm)
        if not any(s.end >= e.start - margin and s.start <= e.end + margin for e in events)
    ]
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return {
        "events": len(events),
        "found": found,
        "delays_s": [round(d, 2) for d in delays],
        "false_alarms": len(false_spans),
        "false_alarm_s": round(sum(s.end - s.start + step for s in false_spans), 2),
        "duration_s": round(times[-1] - times[0] + step, 2) if times else 0.0,
    }


FRAME = re.compile(r"^(?P<clip>.+)_f(?P<frame>\d+)$")


def labelled_times(labels_dir: Path, clip: str, fps: float) -> tuple[list[float], list[float]]:
    with_amb, without = [], []
    for f in labels_dir.glob(f"{clip}_f*.txt"):
        m = FRAME.match(f.stem)
        if not m or m["clip"] != clip:
            continue
        t = int(m["frame"]) / fps
        has = any(line.split()[0] == "0" for line in f.read_text().splitlines() if line.strip())
        (with_amb if has else without).append(t)
    return with_amb, without


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--labels", type=Path, help="folder of labelled frames; without it every video counts as ambulance-free")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--buffer", type=int, default=12, help="read frames in the smoother (12 x stride 2 = about 1 s at 25 fps)")
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="JSON file for the results")
    args = parser.parse_args()

    from ultralytics import YOLO

    from .evaluate import class_index
    from .mine import detect

    model = YOLO(str(args.weights))
    class_id = class_index(model.names, "ambulance")
    results = {}
    for video in args.videos:
        import cv2

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        numbers, frames = detect(video, model, class_id, args.stride, args.imgsz, args.conf)
        times = [n / fps for n in numbers]
        raw = [len(f.scores) > 0 for f in frames]
        alarm = smooth(raw, args.buffer, args.min_frames, args.ratio)
        events = events_from_labels(*labelled_times(args.labels, video.stem, fps), args.merge) if args.labels else []
        results[video.stem] = {
            "smoothed": score_video(events, times, alarm, args.margin),
            "per_frame": score_video(events, times, raw, args.margin),
        }
        print(video.stem, json.dumps(results[video.stem]))
    total = {k: sum(r["smoothed"][k] for r in results.values()) for k in ("events", "found", "false_alarms", "false_alarm_s", "duration_s")}
    raw_total = {k: sum(r["per_frame"][k] for r in results.values()) for k in ("found", "false_alarms")}
    hours = total["duration_s"] / 3600
    summary = {
        "settings": {k: getattr(args, k) for k in ("conf", "stride", "buffer", "min_frames", "ratio", "merge", "margin", "imgsz")},
        "smoothed": total | {"false_alarms_per_hour": round(total["false_alarms"] / hours, 1) if hours else None},
        "per_frame": raw_total,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "videos": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

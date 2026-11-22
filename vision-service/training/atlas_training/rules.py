"""Compare track decision rules on saved runs, without running the models again.

    python -m atlas_training.rules runs/eval/tracks/g_dev.json runs/eval/tracks/g_sariyer.json \
        --videos D:/code/atlas-data/clips/... --out runs/eval/tracks/rules_dev_sariyer.json

Reads the JSON written by atlas_training.tracks (per-frame track probabilities, times and
boxes) and scores a fixed list of rules on every run in it:

- frames: mean p >= threshold over at least n frames (the rule used so far);
- time:   the same, but the track must have been seen for at least s seconds;
- sprt:   sequential test with clipped log-odds (alarm_sprt);
- place:  the frames rule, but a place alarms once (alarm_once_per_place);
- lights: the frames rule, stricter for tracks without flashing red/blue light
          (atlas_training.lights; only for runs that saved light fractions);
the first three each with and without the motion rule (only tracks whose box centre moved at least
`min_move` box widths can alarm).

Runs with labels give passages and clips found; runs without labels (long false-alarm
recordings) give alarm spans and spans per hour. Choose on the development set and the
Sariyer recording only; apply the chosen rule to test2 and İzmir once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lights import alarm_with_lights
from .events import Event, alarm_spans, events_from_labels, score_video, shot_cuts, split_at_cuts
from .tracks import alarm_from_tracks, alarm_from_tracks_time, alarm_once_per_place, alarm_sprt, only_moving

RULES: list[tuple[str, dict]] = (
    [("frames", {"n": 5, "threshold": t}) for t in (0.5, 0.6, 0.7)]
    + [("time", {"seconds": s, "threshold": t}) for s in (0.5, 1.0, 2.0) for t in (0.5, 0.6)]
    + [("sprt", {"upper": u}) for u in (3.0, 5.0)]
    + [("place", {"n": 5, "threshold": t, "radius": 0.5, "window": 600.0}) for t in (0.5, 0.6)]
    + [("lights", {"n": 5, "threshold": 0.5, "strict": s, "min_flicker": f})
       for s in (0.7, 0.8, 1.01) for f in (0.003, 0.01, 0.03)]
)
MIN_MOVES = (None, 0.5, 1.0)


def decide(rule: str, params: dict, times: list[float], frames: list[dict], boxes: list[dict],
           lights: list[dict]) -> list[bool]:
    if rule == "frames":
        return alarm_from_tracks(frames, params["n"], params["threshold"])
    if rule == "time":
        return alarm_from_tracks_time(times, frames, params["seconds"], params["threshold"])
    if rule == "sprt":
        return alarm_sprt(frames, params["upper"])
    if rule == "place":
        return alarm_once_per_place(times, frames, boxes, params["n"], params["threshold"],
                                    params["radius"], params["window"])
    if rule == "lights":
        return alarm_with_lights(times, frames, lights, params["n"], params["threshold"],
                                 params["strict"], params["min_flicker"])
    raise ValueError(rule)


def events_of(run: dict, cuts: list[float], merge: float) -> tuple[list[Event], list[list[float]]] | None:
    """Labelled passages and ignored intervals of a run, or None for an unlabelled one."""
    if "intervals" in run:
        return [Event(a, b) for a, b in run["intervals"]["positive"]], run["intervals"]["ignore"]
    if run.get("with_ambulance") or run.get("without"):
        return split_at_cuts(events_from_labels(run["with_ambulance"], run["without"], merge), cuts), []
    return None


def score_rule(runs: dict, cuts: dict, rule: str, params: dict, min_move: float | None,
               merge: float = 3.0, margin: float = 1.0) -> dict:
    total = {"events": 0, "found": 0, "false_spans": 0, "pos_clips": 0, "pos_found": 0,
             "neg_clips": 0, "neg_alarm": 0, "fa_spans": 0, "fa_hours": 0.0}
    per_run = {}
    for name, run in runs.items():
        times = run["times"]
        frames = [{int(k): v for k, v in f.items()} for f in run["frames"]]
        boxes = [{int(k): v for k, v in b.items()} for b in run["boxes"]]
        if min_move is not None:
            frames = only_moving(frames, boxes, min_move)
        lights = [{int(k): v for k, v in g.items()} for g in run.get("lights", [{}] * len(times))]
        alarm = decide(rule, params, times, frames, boxes, lights)
        labelled = events_of(run, cuts.get(name, []), merge)
        if labelled is None:
            spans = len(alarm_spans(times, alarm))
            hours = (times[-1] - times[0]) / 3600 if len(times) > 1 else 0.0
            total["fa_spans"] += spans
            total["fa_hours"] += hours
            per_run[name] = {"alarm_spans": spans, "hours": round(hours, 2)}
            continue
        events, ignored = labelled
        alarm = [a and not any(lo <= t <= hi for lo, hi in ignored) for t, a in zip(times, alarm, strict=True)]
        s = score_video(events, times, alarm, margin)
        total["events"] += s["events"]
        total["found"] += s["found"]
        total["false_spans"] += s["false_alarms"]
        if events:
            total["pos_clips"] += 1
            total["pos_found"] += s["found"] > 0
        else:
            total["neg_clips"] += 1
            total["neg_alarm"] += any(alarm)
        per_run[name] = {"found": s["found"], "events": s["events"], "false_spans": s["false_alarms"]}
    total["per_run"] = per_run
    return total


def label(rule: str, params: dict, min_move: float | None) -> str:
    p = ",".join(f"{k}={v}" for k, v in params.items())
    return f"{rule}({p})" + (f"+move{min_move}" if min_move is not None else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="JSON files written by atlas_training.tracks")
    parser.add_argument("--videos", nargs="*", type=Path, default=[], help="videos of labelled runs, for shot cuts")
    parser.add_argument("--cuts", type=Path, help="cache of shot cuts (JSON), read and updated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs: dict = {}
    for path in args.runs:
        runs.update(json.loads(path.read_text(encoding="utf-8"))["runs"])
    cuts = json.loads(args.cuts.read_text(encoding="utf-8")) if args.cuts and args.cuts.exists() else {}
    for video in args.videos:
        if video.stem in runs and video.stem not in cuts:
            cuts[video.stem] = shot_cuts(video)
    if args.cuts:
        args.cuts.write_text(json.dumps(cuts) + "\n", encoding="utf-8")

    rows = []
    for rule, params in RULES:
        if rule == "lights" and not all("lights" in r for r in runs.values()):
            continue
        for min_move in MIN_MOVES if rule not in ("place", "lights") else (None,):
            r = score_rule(runs, cuts, rule, params, min_move)
            r["rule"] = label(rule, params, min_move)
            rows.append(r)
            fa = f" | unlabelled: {r['fa_spans']} spans in {r['fa_hours']:.2f} h" if r["fa_hours"] else ""
            print(f"{r['rule']:<38} passages {r['found']}/{r['events']}, clips {r['pos_found']}/{r['pos_clips']}, "
                  f"negative clips with alarm {r['neg_alarm']}/{r['neg_clips']}, false spans {r['false_spans']}{fa}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

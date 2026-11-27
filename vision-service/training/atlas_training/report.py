"""Apply the chosen track rule once to held-out runs and print the NFR-04 / NFR-05 numbers.

    python -m atlas_training.report runs/eval/tracks/g5g_test2.json runs/eval/tracks/t3_v5g.json \
        --fa runs/eval/tracks/g5g_izmir_day.json runs/eval/tracks/g5g_izmir_night.json \
        --out runs/eval/report_v5g.json

The rule is fixed here (chosen on the development set and the Sarıyer camera only, see
RESEARCH.md); this tool never searches over rules, so the held-out numbers stay honest.
Recall is the share of ambulance videos with at least one passage found; precision is the
share of alarm spans that fall on a labelled passage (found passages / (found + false
spans)); false alarms per hour come from the unlabelled camera recordings, with a 95 %
Poisson upper bound. Intervals are 95 % Wilson intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import false_alarm_upper_bound, wilson_interval
from .rules import score_rule

CHOSEN = ("place", {"n": 5, "threshold": 0.7, "radius": 0.5, "window": 1800.0})


def load(paths: list[Path], intervals: Path | None = None) -> dict:
    """Runs of atlas_training.tracks; with `intervals`, their labels are replaced by that
    file's (e.g. an audited copy), without running the models again."""
    runs: dict = {}
    for path in paths:
        runs.update(json.loads(path.read_text(encoding="utf-8"))["runs"])
    if intervals:
        spec = json.loads(intervals.read_text(encoding="utf-8"))["videos"]
        for name, run in runs.items():
            if name in spec and "intervals" in run:
                run["intervals"] = {"positive": spec[name].get("positive", []),
                                    "ignore": spec[name].get("ignore", [])}
    return runs


def reviewed_false_alarms(runs: dict, review: Path, tolerance: float = 2.0) -> dict:
    """Alarm spans of unlabelled recordings, split by an eye check (run,start_s,verdict).

    A camera recording assumed to be ambulance-free can still show a real ambulance (the
    İzmir square did). Each alarm span is looked at; spans marked "ambulance" are true
    detections, and every span not marked (unchecked) counts as false, to stay on the
    safe side. Spans of the same track a moment apart are also counted as events."""
    import csv

    from .events import alarm_spans
    from .rules import decide

    verdicts = [(r["run"], float(r["start_s"]), r["verdict"])
                for r in csv.DictReader(review.open(encoding="utf-8"))]
    out = {"true": 0, "false": 0, "unchecked": 0, "hours": 0.0, "spans": []}
    rule, params = CHOSEN
    for name, run in runs.items():
        times = run["times"]
        frames = [{int(k): v for k, v in f.items()} for f in run["frames"]]
        boxes = [{int(k): v for k, v in b.items()} for b in run["boxes"]]
        alarm = decide(rule, params, times, frames, boxes, [{}] * len(times))
        out["hours"] += (times[-1] - times[0]) / 3600 if len(times) > 1 else 0.0
        for span in alarm_spans(times, alarm):
            match = [v for r, t, v in verdicts if r == name and abs(t - span.start) <= tolerance]
            verdict = match[0] if match else "unchecked"
            out["true" if verdict == "ambulance" else "false"] += 1
            out["unchecked"] += verdict == "unchecked"
            out["spans"].append([name, round(span.start, 1), verdict])
    out["per_hour"] = out["false"] / out["hours"] if out["hours"] else 0.0
    out["upper_bound"] = false_alarm_upper_bound(out["false"], out["hours"])
    return out


def summarise(r: dict) -> dict:
    alarms_true, alarms_false = r["found"], r["false_spans"]
    out = {
        "passages": [r["found"], r["events"]],
        "recall_videos": [r["pos_found"], r["pos_clips"], *wilson_interval(r["pos_found"], r["pos_clips"])],
        "precision_alarms": [alarms_true, alarms_true + alarms_false,
                             *wilson_interval(alarms_true, alarms_true + alarms_false)],
        "negative_videos_with_alarm": [r["neg_alarm"], r["neg_clips"]],
    }
    if r["fa_hours"]:
        out["false_alarms"] = [r["fa_spans"], round(r["fa_hours"], 2), r["fa_spans"] / r["fa_hours"],
                               false_alarm_upper_bound(r["fa_spans"], r["fa_hours"])]
    out["per_run"] = r["per_run"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labelled", nargs="*", type=Path, help="tracks JSON of labelled held-out videos")
    parser.add_argument("--fa", nargs="*", type=Path, default=[], help="tracks JSON of unlabelled camera recordings")
    parser.add_argument("--intervals", type=Path, help="labels to score the labelled runs with (overrides those saved in them)")
    parser.add_argument("--reviewed", type=Path, help="eye check of camera alarms (run,start_s,verdict,note)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rule, params = CHOSEN
    result = {"rule": [rule, params]}
    for name, paths in (("labelled", args.labelled), ("false_alarm", args.fa)):
        if paths:
            result[name] = summarise(score_rule(load(paths, args.intervals if name == "labelled" else None), {}, rule, params, None))
    for name in ("labelled", "false_alarm"):
        s = result.get(name)
        if not s:
            continue
        if name == "labelled":
            k, n, lo, hi = s["recall_videos"]
            print(f"ambulance videos found {k}/{n} ({lo:.0%}-{hi:.0%}), passages {s['passages'][0]}/{s['passages'][1]}")
            k, n, lo, hi = s["precision_alarms"]
            print(f"alarms on an ambulance {k}/{n} ({lo:.0%}-{hi:.0%}), "
                  f"ambulance-free videos with an alarm {s['negative_videos_with_alarm'][0]}/{s['negative_videos_with_alarm'][1]}")
        else:
            a, h, rate, up = s["false_alarms"]
            print(f"alarm spans {a} in {h} h = {rate:.2f}/h (95 % upper bound {up:.2f}/h), before the eye check")
    if args.fa and args.reviewed:
        r = result["reviewed"] = reviewed_false_alarms(load(args.fa), args.reviewed)
        print(f"after the eye check: {r['true']} spans on a real ambulance, {r['false']} false "
              f"({r['unchecked']} unchecked, counted false) in {r['hours']:.2f} h = {r['per_hour']:.2f}/h "
              f"(95 % upper bound {r['upper_bound']:.2f}/h)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Compare crop classifiers on a fixed, eye-checked set of hard tracks, with intervals.

    python -m atlas_training.bench D:/code/atlas-data/labeling/bench_v1 D:/code/atlas-data/labeling/bench_v1_sariyer \
        --classifiers v5g=runs/cls/crops_v5g/weights/best.pt v5i=runs/cls/crops_v5i/weights/best.pt \
        --out runs/eval/bench_v1.json

Why: on the development videos a classifier raises only a handful of alarms, so two
classifiers (or two training runs of the same one) differ by one or two alarms, which is
noise (see RESEARCH.md, second seed). This set holds many tracks that look like an
ambulance, each one checked by eye (verdicts.csv of atlas_training.trackmine: ambulance,
other, skip), from the development videos and the Sarıyer camera only; the test sets stay
untouched. Every crop is the 224 px input the classifier sees, so a classifier is scored
offline in seconds.

For each classifier:
- the share of `other` tracks that the track rule would raise (mean p >= --threshold over
  >= --min-frames crops): false-alarm tracks, the number to push down;
- the share of `ambulance` tracks it raises: recall on hard ambulance tracks;
- crop-level rates at p >= 0.5, for reference.
Intervals are 95 % percentile bootstrap over tracks (crops of one track are near copies,
so tracks are resampled, not crops). For every pair of classifiers the difference is
bootstrapped on the same resampled tracks (paired), so "B is better than A" is claimed
only when the interval of the difference excludes 0.

Caveat: tracks were proposed by one classifier (those with some crop at p >= the mining
threshold), so the set over-represents that classifier's mistakes; mine with a low
threshold and report which classifier proposed the tracks.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_tracks(roots: list[Path], max_crops: int = 0) -> list[dict]:
    """[{"id", "label", "files"}] for checked tracks; skip tracks are left out. With
    max_crops, a longer track (a vehicle parked for an hour has thousands of crops) keeps
    that many crops spread evenly over it, which keeps its mean and makes scoring fast."""
    tracks = []
    for root in roots:
        for v in csv.DictReader((root / "verdicts.csv").open(encoding="utf-8")):
            if v["verdict"] not in ("ambulance", "other"):
                continue
            lo = float(v["from_s"]) if v.get("from_s") else -1.0
            hi = float(v["until_s"]) if v.get("until_s") else 1e9
            files = [f for f in sorted((root / v["video"] / v["track"]).glob("*.jpg"))
                     if lo <= float(f.stem.split("_")[0]) <= hi]
            if max_crops and len(files) > max_crops:
                files = [files[int(i)] for i in np.linspace(0, len(files) - 1, max_crops)]
            if files:
                tracks.append({"id": f"{root.name}/{v['video']}/{v['track']}", "label": v["verdict"], "files": files})
    return tracks


def track_raised(p: np.ndarray, min_frames: int, threshold: float) -> bool:
    """The track rule on a whole track: enough crops and a high mean."""
    return len(p) >= min_frames and float(np.mean(p)) >= threshold


def rates(tracks: list[dict], probs: dict[str, np.ndarray], idx: np.ndarray, min_frames: int,
          threshold: float) -> dict[str, float]:
    """Rates over the tracks at positions idx (a bootstrap sample may repeat tracks)."""
    out = {}
    for label, key in (("other", "false_alarm_tracks"), ("ambulance", "recall_tracks")):
        sel = [i for i in idx if tracks[i]["label"] == label]
        out[key] = float(np.mean([track_raised(probs[tracks[i]["id"]], min_frames, threshold) for i in sel])) if sel else float("nan")
        crops = np.concatenate([probs[tracks[i]["id"]] for i in sel]) if sel else np.array([])
        out[key.replace("tracks", "crops")] = float(np.mean(crops >= 0.5)) if len(crops) else float("nan")
    return out


def bootstrap(tracks: list[dict], all_probs: dict[str, dict[str, np.ndarray]], min_frames: int, threshold: float,
              n: int = 2000, seed: int = 0) -> dict:
    """Point values, 95 % intervals per classifier, and paired differences between classifiers."""
    rng = np.random.default_rng(seed)
    names = list(all_probs)
    full = np.arange(len(tracks))
    point = {m: rates(tracks, all_probs[m], full, min_frames, threshold) for m in names}
    samples = {m: [] for m in names}
    # Resample within each class, so every sample has both kinds of track.
    by_label = {lab: np.array([i for i, t in enumerate(tracks) if t["label"] == lab]) for lab in ("ambulance", "other")}
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_label.values() if len(ix)])
        for m in names:
            samples[m].append(rates(tracks, all_probs[m], idx, min_frames, threshold))
    out = {"classifiers": {}, "pairs": {}}
    for m in names:
        out["classifiers"][m] = {k: {"value": point[m][k], "ci": [float(np.nanpercentile([s[k] for s in samples[m]], 2.5)),
                                                               float(np.nanpercentile([s[k] for s in samples[m]], 97.5))]}
                                 for k in point[m]}
    for a, b in combinations(names, 2):
        pair = {}
        for k in point[a]:
            d = np.array([sb[k] - sa[k] for sa, sb in zip(samples[a], samples[b], strict=True)])
            lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
            pair[k] = {"diff": point[b][k] - point[a][k], "ci": [lo, hi], "clear": bool(lo > 0 or hi < 0)}
        out["pairs"][f"{b} - {a}"] = pair
    return out


def score(tracks: list[dict], weights: Path, batch: int = 128, flip: bool = False) -> dict[str, np.ndarray]:
    """p(ambulance) for every crop; with flip, the mean of the crop and its mirror image."""
    from ultralytics import YOLO

    from .evaluate import class_index

    model = YOLO(str(weights))
    amb = class_index(model.names, "ambulance")
    files = [f for t in tracks for f in t["files"]]
    ps = []
    for i in range(0, len(files), batch):
        chunk = [str(f) for f in files[i:i + batch]]
        p = np.array([r.probs.data[amb].item() for r in model.predict(chunk, imgsz=224, verbose=False)])
        if flip:
            import cv2

            mirrored = [cv2.flip(cv2.imread(f), 1) for f in chunk]
            p = (p + np.array([r.probs.data[amb].item() for r in model.predict(mirrored, imgsz=224, verbose=False)])) / 2
        ps += p.tolist()
    out, k = {}, 0
    for t in tracks:
        out[t["id"]] = np.array(ps[k:k + len(t["files"])])
        k += len(t["files"])
    return out


def score_spec(tracks: list[dict], spec: str, cache: dict) -> dict[str, np.ndarray]:
    """Scores for a classifier spec: `a.pt`, `a.pt+b.pt` (mean of the members' p) and an
    optional `@flip` suffix (mirror-image test-time augmentation for every member)."""
    flip = spec.endswith("@flip")
    members = spec.removesuffix("@flip").split("+")
    runs = []
    for w in members:
        key = (w, flip)
        if key not in cache:
            cache[key] = score(tracks, Path(w), flip=flip)
        runs.append(cache[key])
    return {t: np.mean([r[t] for r in runs], axis=0) for t in runs[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", type=Path, nargs="+", help="trackmine folders with a checked verdicts.csv")
    parser.add_argument("--classifiers", nargs="+", required=True, help="name=weights.pt, name=a.pt+b.pt (ensemble), name=a.pt@flip (mirror TTA)")
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-crops", type=int, default=60, help="crops kept per track, spread evenly (0 = all)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tracks = load_tracks(args.roots, args.max_crops)
    counts = {lab: sum(t["label"] == lab for t in tracks) for lab in ("ambulance", "other")}
    crops = sum(len(t["files"]) for t in tracks)
    print(f"{len(tracks)} tracks ({counts}), {crops} crops")
    all_probs, cache = {}, {}
    for item in args.classifiers:
        name, spec = item.split("=", 1)
        all_probs[name] = score_spec(tracks, spec, cache)
    result = bootstrap(tracks, all_probs, args.min_frames, args.threshold, args.bootstrap)
    result.update({"tracks": counts, "crops": crops, "min_frames": args.min_frames, "threshold": args.threshold,
                   "roots": [str(r) for r in args.roots]})
    for m, r in result["classifiers"].items():
        fa, rc = r["false_alarm_tracks"], r["recall_tracks"]
        print(f"{m:10} false-alarm tracks {fa['value']:.3f} [{fa['ci'][0]:.3f}, {fa['ci'][1]:.3f}]  "
              f"recall tracks {rc['value']:.3f} [{rc['ci'][0]:.3f}, {rc['ci'][1]:.3f}]  "
              f"crops: FP {r['false_alarm_crops']['value']:.3f}, recall {r['recall_crops']['value']:.3f}")
    for pair, r in result["pairs"].items():
        fa, rc = r["false_alarm_tracks"], r["recall_tracks"]
        print(f"{pair:22} false-alarm tracks {fa['diff']:+.3f} [{fa['ci'][0]:+.3f}, {fa['ci'][1]:+.3f}]{' *' if fa['clear'] else ''}  "
              f"recall tracks {rc['diff']:+.3f} [{rc['ci'][0]:+.3f}, {rc['ci'][1]:+.3f}]{' *' if rc['clear'] else ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Build a classifier crop set from an existing one plus checked, mined tracks.

    python -m atlas_training.buildmined datasets/crops_v5e D:/code/atlas-data/labeling/trackmine_v1 \
        --out datasets/crops_v5g [--night 0.3]

Reads verdicts.csv of atlas_training.trackmine (video, track, verdict, from_s, until_s):

- ambulance tracks: crops inside [from_s, until_s], at most --max-pos per track, the
  lowest-scoring first (the views the classifier does not know yet), plus one degraded
  low-resolution copy each (atlas_training.lowres.degrade, width 32-80 px) for small,
  far ambulances;
- other tracks: crops inside the range with p >= --neg-min-p (vehicles that fooled the
  classifier), at most --max-neg per track;
- skip tracks are not used.

Mined crops go to train only; val and test are copied unchanged, so the new classifier is
compared with the old one on the same validation crops (and on the development videos).
With --night, that share of the ambulance training crops also gets a synthetic night copy
(atlas_training.night), to be compared with a set built without it.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from .lowres import degrade
from .night import nightify


def crops_of(root: Path, video: str, track: str, lo: float, hi: float) -> list[tuple[float, Path]]:
    out = []
    for f in sorted((root / video / track).glob("*.jpg")):
        t, p = (float(v) for v in f.stem.split("_"))
        if lo <= t <= hi:
            out.append((p, f))
    return out


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="existing crop set (train/val/test, ambulance/other)")
    parser.add_argument("mined", type=Path, nargs="+", help="trackmine output folders with verdicts.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pos", type=int, default=40)
    parser.add_argument("--max-neg", type=int, default=30)
    parser.add_argument("--neg-min-p", type=float, default=0.2)
    parser.add_argument("--night", type=float, default=0.0, help="share of ambulance train crops given a night copy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name")
    shutil.copytree(args.base, args.out, ignore=shutil.ignore_patterns("*.cache"))
    counts = {"pos": 0, "pos_lowres": 0, "neg": 0, "night": 0, "tracks_pos": 0, "tracks_neg": 0}
    for root in args.mined:
        for v in csv.DictReader((root / "verdicts.csv").open(encoding="utf-8")):
            if v["verdict"] not in ("ambulance", "other"):
                continue
            lo = float(v["from_s"]) if v["from_s"] else -1.0
            hi = float(v["until_s"]) if v["until_s"] else 1e9
            items = crops_of(root, v["video"], v["track"], lo, hi)
            prefix = f"mined__{v['video']}__{v['track']}"
            if v["verdict"] == "ambulance":
                items.sort(key=lambda x: x[0])
                chosen = items[: args.max_pos]
                folder = args.out / "train" / "ambulance"
                for k, (p, f) in enumerate(chosen):
                    image = cv2.imread(str(f))
                    cv2.imwrite(str(folder / f"{prefix}_{k:03d}.jpg"), image)
                    width = int(rng.integers(32, 81))
                    cv2.imwrite(str(folder / f"{prefix}_{k:03d}_lr{width}.jpg"), degrade(image, width))
                counts["pos"] += len(chosen)
                counts["pos_lowres"] += len(chosen)
                counts["tracks_pos"] += bool(chosen)
            else:
                hard = [x for x in items if x[0] >= args.neg_min_p]
                if len(hard) > args.max_neg:
                    hard = [hard[i] for i in rng.choice(len(hard), args.max_neg, replace=False)]
                folder = args.out / "train" / "other"
                for k, (p, f) in enumerate(hard):
                    shutil.copy(f, folder / f"{prefix}_{k:03d}.jpg")
                counts["neg"] += len(hard)
                counts["tracks_neg"] += bool(hard)
    if args.night > 0:
        folder = args.out / "train" / "ambulance"
        originals = [f for f in sorted(folder.glob("*.jpg")) if "_lr" not in f.stem and "_night" not in f.stem]
        for f in originals:
            if rng.random() < args.night:
                cv2.imwrite(str(f.with_name(f.stem + "_night.jpg")), nightify(cv2.imread(str(f)), rng))
                counts["night"] += 1
    totals = {c: len(list((args.out / "train" / c).glob("*.jpg"))) for c in ("ambulance", "other")}
    print(counts, "train totals:", totals)


if __name__ == "__main__":
    main()

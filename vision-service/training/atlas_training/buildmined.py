"""Build a classifier crop set from an existing one plus checked, mined tracks.

    python -m atlas_training.buildmined datasets/crops_v5e D:/code/atlas-data/labeling/trackmine_v1 \
        --out datasets/crops_v5g [--night 0.3]

Reads verdicts.csv of atlas_training.trackmine (video, track, verdict, from_s, until_s):

- ambulance tracks: crops inside [from_s, until_s], at most --max-pos per track: half the
  lowest-scoring (the views the classifier does not know yet), half spread over the track, plus one degraded
  low-resolution copy each (atlas_training.lowres.degrade, width 32-80 px) for small,
  far ambulances;
- other tracks: crops inside the range with p >= --neg-min-p (vehicles that fooled the
  classifier), at most --max-neg per track;
- skip tracks are not used.

Mined crops go to train only; val and test are copied unchanged, so the new classifier is
compared with the old one on the same validation crops (and on the development videos).
Before building, --preview writes numbered contact sheets of the crops that could be
wrong: chosen positives with p < --check-below (a track that jumped to another vehicle
scores low there) and every chosen negative (a jump to an ambulance scores high). Crops
found wrong are listed as "sheet:tile" lines in --exclude and left out.

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


def select(args, rng) -> list[tuple[str, float, Path, str]]:
    """(kind, p, source crop, output name) of every crop the verdicts allow."""
    chosen = []
    for root in args.mined:
        for v in csv.DictReader((root / "verdicts.csv").open(encoding="utf-8")):
            if v["verdict"] not in ("ambulance", "other"):
                continue
            lo = float(v["from_s"]) if v["from_s"] else -1.0
            hi = float(v["until_s"]) if v["until_s"] else 1e9
            items = crops_of(root, v["video"], v["track"], lo, hi)
            prefix = f"mined__{v['video']}__{v['track']}"
            if v["verdict"] == "ambulance":
                # Half the lowest-scoring crops (views the classifier misses), half spread
                # evenly over the rest of the track (consecutive frames are near copies).
                low = sorted(range(len(items)), key=lambda i: items[i][0])[: args.max_pos // 2]
                rest = [i for i in range(len(items)) if i not in set(low)]
                spread = [rest[int(j)] for j in np.linspace(0, len(rest) - 1, min(len(rest), args.max_pos - len(low)))] if rest else []
                for k, i in enumerate(sorted(set(low) | set(spread))):
                    p, f = items[i]
                    chosen.append(("pos", p, f, f"{prefix}_{k:03d}"))
            else:
                hard = [x for x in items if x[0] >= args.neg_min_p]
                if len(hard) > args.max_neg:
                    hard = [hard[i] for i in sorted(rng.choice(len(hard), args.max_neg, replace=False))]
                for k, (p, f) in enumerate(hard):
                    chosen.append(("neg", p, f, f"{prefix}_{k:03d}"))
    return chosen


def to_check(chosen, below: float):
    return [c for c in chosen if c[0] == "neg" or c[1] < below]


def preview(chosen, args) -> None:
    """Numbered sheets (10 x 10 tiles) of the crops to check, and index.csv."""
    import cv2

    items = to_check(chosen, args.check_below)
    args.preview.mkdir(parents=True, exist_ok=True)
    with (args.preview / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "tile", "kind", "p", "crop"])
        for start in range(0, len(items), 100):
            sheet = start // 100
            tiles = []
            for n, (kind, p, crop, _) in enumerate(items[start:start + 100]):
                tile = cv2.resize(cv2.imread(str(crop)), (128, 128))
                colour = (0, 255, 0) if kind == "pos" else (0, 0, 255)
                cv2.rectangle(tile, (0, 0), (127, 127), colour, 2)
                cv2.putText(tile, f"{n} {kind} {p:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                tiles.append(tile)
                w.writerow([sheet, n, kind, f"{p:.3f}", crop])
            tiles += [np.zeros((128, 128, 3), np.uint8)] * (-len(tiles) % 10)
            grid = np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)])
            cv2.imwrite(str(args.preview / f"check_{sheet:03d}.jpg"), grid)
    pos = sum(c[0] == "pos" for c in chosen)
    print(f"{len(chosen)} crops chosen ({pos} positive); {len(items)} to check in {args.preview}")


def load_excluded(args) -> set[str]:
    """Source paths listed as sheet:tile in each --exclude file (next to its index.csv)."""
    out = set()
    for path in args.exclude or []:
        index = {(r["sheet"], r["tile"]): r["crop"]
                 for r in csv.DictReader((path.parent / "index.csv").open(encoding="utf-8"))}
        for line in path.read_text(encoding="utf-8").split():
            sheet, tile = line.strip().split(":")
            out.add(index[(str(int(sheet)), str(int(tile)))])
    return out


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="existing crop set (train/val/test, ambulance/other)")
    parser.add_argument("mined", type=Path, nargs="+", help="trackmine output folders with verdicts.csv")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preview", type=Path, help="write sheets of crops to check here, build nothing")
    parser.add_argument("--check-below", type=float, default=0.3, help="positives below this p are checked")
    parser.add_argument("--exclude", type=Path, nargs="+", help="sheet:tile lines of crops found wrong (next to index.csv)")
    parser.add_argument("--max-pos", type=int, default=16)
    parser.add_argument("--max-neg", type=int, default=30)
    parser.add_argument("--neg-min-p", type=float, default=0.2)
    parser.add_argument("--night", type=float, default=0.0, help="share of ambulance train crops given a night copy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    chosen = select(args, rng)
    if args.preview:
        preview(chosen, args)
        return
    excluded = load_excluded(args)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name")
    shutil.copytree(args.base, args.out, ignore=shutil.ignore_patterns("*.cache"))
    counts = {"pos": 0, "pos_lowres": 0, "neg": 0, "night": 0, "excluded": 0}
    for kind, p, f, name in chosen:
        if str(f) in excluded:
            counts["excluded"] += 1
            continue
        if kind == "pos":
            image = cv2.imread(str(f))
            folder = args.out / "train" / "ambulance"
            cv2.imwrite(str(folder / f"{name}.jpg"), image)
            width = int(rng.integers(32, 81))
            cv2.imwrite(str(folder / f"{name}_lr{width}.jpg"), degrade(image, width))
            counts["pos"] += 1
            counts["pos_lowres"] += 1
        else:
            shutil.copy(f, args.out / "train" / "other" / f"{name}.jpg")
            counts["neg"] += 1
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

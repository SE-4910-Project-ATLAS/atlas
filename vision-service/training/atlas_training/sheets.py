"""Contact sheets of mined tracks that still need a verdict (atlas_training.trackmine).

    python -m atlas_training.sheets D:/code/atlas-data/labeling/trackmine_v1 --min-max-p 0.3

Shows, per video, only the tracks whose highest classifier score reached --min-max-p and
that have no line in verdicts.csv yet, at most --per-sheet tracks per image (readable size).
Low-scoring tracks of other vehicles teach the classifier nothing new and are not shown.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_track(folder: Path) -> list[tuple[float, float, Path]]:
    items = []
    for f in sorted(folder.glob("*.jpg")):
        t, p = f.stem.split("_")
        items.append((float(t), float(p), f))
    return items


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-max-p", type=float, default=0.3)
    parser.add_argument("--per-sheet", type=int, default=12)
    parser.add_argument("--per-track", type=int, default=10)
    args = parser.parse_args()

    done = set()
    verdicts = args.root / "verdicts.csv"
    if verdicts.exists():
        done = {(r["video"], r["track"]) for r in csv.DictReader(verdicts.open(encoding="utf-8"))}
    # Read the track folders directly, so finished videos can be reviewed while mining runs
    # (a video's folders are complete once its sheet in sheets/ exists).
    finished = {f.stem for f in (args.root / "sheets").glob("*.jpg")}
    rows = []
    for video in sorted(finished):
        for folder in sorted((args.root / video).glob("*"), key=lambda f: int(f.name) if f.name.isdigit() else 0):
            items = load_track(folder)
            if not items or (video, folder.name) in done:
                continue
            max_p = max(p for _, p, _ in items)
            if max_p >= args.min_max_p:
                rows.append({"video": video, "track": folder.name, "frames": len(items), "max_p": f"{max_p:.2f}"})
    out = args.root / "review"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for start in range(0, len(rows), args.per_sheet):
        strips = []
        for r in rows[start:start + args.per_sheet]:
            items = load_track(args.root / r["video"] / r["track"])
            pick = [items[i] for i in np.linspace(0, len(items) - 1, min(args.per_track, len(items))).astype(int)]
            tiles = []
            for t, p, f in pick:
                tile = cv2.resize(cv2.imread(str(f)), (112, 112))
                cv2.putText(tile, f"{p:.2f} {t:.0f}s", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                tiles.append(tile)
            tiles += [np.zeros((112, 112, 3), np.uint8)] * (args.per_track - len(tiles))
            label = np.zeros((112, 170, 3), np.uint8)
            cv2.putText(label, r["video"][:11], (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(label, f"track {r['track']}", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(label, f"n={r['frames']} max={r['max_p']}", (4, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            strips.append(np.hstack([label] + tiles))
        cv2.imwrite(str(out / f"review_{start // args.per_sheet:03d}.jpg"), np.vstack(strips))
    print(f"{len(rows)} tracks to review in {out}")


if __name__ == "__main__":
    main()

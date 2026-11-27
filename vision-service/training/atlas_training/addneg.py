"""Add checked negative crops (e.g. vans from a public dataset) to a copy of a crop set.

    python -m atlas_training.addneg datasets/crops_v5g D:/code/atlas-data/labeling/van_2s73c \
        --exclude D:/code/atlas-data/labeling/van_2s73c_exclude.txt --prefix van2s73c --out datasets/crops_v5i

The folder holds 224 px square crops named <p>_<name>.jpg (p = the current classifier's
p(ambulance), written when the crops were made). Vans look like ambulances, and a public
"van" set can contain real ambulances, so every crop the classifier scores high is looked
at first and the ambulances are listed in --exclude (file names, one per line); they are
left out. All others go to train/other only; val and test are copied unchanged, so the new
classifier is compared with the old one on the same validation crops.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def chosen(folder: Path, excluded: set[str], min_p: float) -> list[Path]:
    out = []
    for f in sorted(folder.glob("*.jpg")):
        if f.name in excluded:
            continue
        if float(f.name.split("_", 1)[0]) < min_p:
            continue
        out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="existing crop set (train/val/test, ambulance/other)")
    parser.add_argument("folders", type=Path, nargs="+", help="folders of checked negative crops")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="files listing crops found to be ambulances")
    parser.add_argument("--prefix", nargs="+", required=True, help="one name prefix per folder")
    parser.add_argument("--min-p", type=float, default=0.0, help="only crops the current classifier scores at least this")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.prefix) != len(args.folders):
        raise SystemExit("give one --prefix per folder")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name")
    excluded = {line.strip() for path in args.exclude for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    shutil.copytree(args.base, args.out, ignore=shutil.ignore_patterns("*.cache"))
    target = args.out / "train" / "other"
    for folder, prefix in zip(args.folders, args.prefix, strict=True):
        files = chosen(folder, excluded, args.min_p)
        for f in files:
            shutil.copy(f, target / f"{prefix}__{f.name}")
        print(f"{folder}: {len(files)} crops added, {len(excluded)} excluded names")
    totals = {c: len(list((args.out / "train" / c).glob("*.jpg"))) for c in ("ambulance", "other")}
    print("train totals:", totals)


if __name__ == "__main__":
    main()

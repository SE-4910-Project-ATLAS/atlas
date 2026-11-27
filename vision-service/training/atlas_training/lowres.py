"""Add low-resolution copies of the train crops of a classification set.

    python -m atlas_training.lowres datasets/crops_v5b --out datasets/crops_v5e

Copies the set and, for every train crop, adds one copy shrunk to a random width between
--min-px and --max-px and enlarged back to the crop size, so that the classifier also learns
from crops that look like far vehicles. Mixed-resolution training kept recognition at low
resolution in face recognition (Knoche and Hormann) and multi-label classification
(SpliceMix); zero-shot accuracy of CLIP-type models drops from about 64 px and steeply
below 32 px (LR0.FM, ICLR 2025). Val and test are left as they are.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def degrade(image: np.ndarray, width: int) -> np.ndarray:
    """Shrink to `width` pixels wide (area interpolation) and enlarge back (bilinear)."""
    import cv2

    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, width), max(1, round(width * h / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-px", type=int, default=24)
    parser.add_argument("--max-px", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import cv2

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new folder")
    shutil.copytree(args.source, args.out)
    rng = np.random.default_rng(args.seed)
    n = 0
    for crop in sorted((args.out / "train").glob("*/*.jpg")):
        width = int(rng.integers(args.min_px, args.max_px + 1))
        cv2.imwrite(str(crop.with_name(f"{crop.stem}_lr{width}.jpg")), degrade(cv2.imread(str(crop)), width),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    print(f"{n} low-resolution copies added to {args.out / 'train'}")


if __name__ == "__main__":
    main()

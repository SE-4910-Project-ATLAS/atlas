"""Contact sheet of one video, one frame per --every seconds, for time-interval labels.

    python -m atlas_training.timesheet D:/code/atlas-data/clips/test3/*.mp4 --out D:/code/atlas-data/labeling/test3_sheets

Each tile shows the second it was taken at, so the seconds in which an ambulance is
visible can be written into labels.json ("positive" and "ignore" intervals, the format of
clips/test2/labels.json). Long videos are split over several sheets of --per-sheet tiles.
Used only for labelling held-out test videos, which never go into training or tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=float, default=1.0)
    parser.add_argument("--per-sheet", type=int, default=60)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile", type=int, default=320, help="tile width in pixels")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        tiles = []
        t = 0.0
        while t < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            tile = cv2.resize(frame, (args.tile, round(args.tile * h / w)))
            cv2.rectangle(tile, (0, 0), (52, 20), (0, 0, 0), -1)
            cv2.putText(tile, f"{t:.0f}s", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(tile)
            t += args.every
        capture.release()
        if not tiles:
            continue
        th = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, th - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
        for n, start in enumerate(range(0, len(tiles), args.per_sheet)):
            chunk = tiles[start:start + args.per_sheet]
            chunk += [np.zeros_like(chunk[0])] * (-len(chunk) % args.cols)
            grid = np.vstack([np.hstack(chunk[i:i + args.cols]) for i in range(0, len(chunk), args.cols)])
            cv2.imwrite(str(args.out / f"{video.stem}_{n}.jpg"), grid)
        print(f"{video.stem}: {duration:.0f} s, {len(tiles)} frames", flush=True)


if __name__ == "__main__":
    main()

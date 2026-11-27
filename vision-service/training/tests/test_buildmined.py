import csv
import sys

import cv2
import numpy as np

from atlas_training import buildmined


def _crop(path, value=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))


def test_build_uses_ranges_caps_and_verdicts(tmp_path, monkeypatch):
    base = tmp_path / "base"
    for split in ("train", "val"):
        for cls in ("ambulance", "other"):
            _crop(base / split / cls / "x.jpg")
    mined = tmp_path / "mined"
    for t, p in [(1.0, 0.9), (2.0, 0.1), (3.0, 0.2), (9.0, 0.0)]:
        _crop(mined / "v" / "1" / f"{t:08.2f}_{p:.3f}.jpg")
    for t, p in [(1.0, 0.1), (2.0, 0.8)]:
        _crop(mined / "v" / "2" / f"{t:08.2f}_{p:.3f}.jpg")
    _crop(mined / "v" / "3" / "00001.00_0.900.jpg")
    with (mined / "verdicts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track", "verdict", "from_s", "until_s", "note"])
        w.writerow(["v", "1", "ambulance", "", "5", ""])
        w.writerow(["v", "2", "other", "", "", ""])
        w.writerow(["v", "3", "skip", "", "", ""])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(mined), "--out", str(out), "--max-pos", "2"])
    buildmined.main()
    pos = sorted(f.name for f in (out / "train" / "ambulance").glob("mined__*"))
    # Track 1: 3 crops inside 0-5 s, the 2 lowest-scoring kept, each with a low-resolution copy.
    assert len(pos) == 4 and sum("_lr" in n for n in pos) == 2
    neg = list((out / "train" / "other").glob("mined__*"))
    assert len(neg) == 1          # only the p >= 0.2 crop of track 2
    assert not list((out / "train" / "ambulance").glob("mined__v__3*"))
    assert (out / "val" / "ambulance" / "x.jpg").exists()


def test_preview_then_exclude(tmp_path, monkeypatch):
    base = tmp_path / "base"
    for cls in ("ambulance", "other"):
        _crop(base / "train" / cls / "x.jpg")
    mined = tmp_path / "mined"
    for t, p in [(1.0, 0.05), (2.0, 0.9)]:
        _crop(mined / "v" / "1" / f"{t:08.2f}_{p:.3f}.jpg")
    with (mined / "verdicts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "track", "verdict", "from_s", "until_s", "note"])
        w.writerow(["v", "1", "ambulance", "", "", ""])
    check = tmp_path / "check"
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(mined), "--preview", str(check)])
    buildmined.main()
    rows = list(csv.DictReader((check / "index.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and float(rows[0]["p"]) < 0.3      # only the low-scoring positive is checked
    (check / "exclude.txt").write_text("0:0\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(mined), "--out", str(out), "--exclude", str(check / "exclude.txt")])
    buildmined.main()
    kept = [f.name for f in (out / "train" / "ambulance").glob("mined__*") if "_lr" not in f.name]
    assert len(kept) == 1                                     # the 0.9 crop stays, the excluded one is gone

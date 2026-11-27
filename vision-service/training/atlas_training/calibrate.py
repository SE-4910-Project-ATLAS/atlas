"""Calibrate the crop classifier on val and choose the track threshold there, not on test.

    python -m atlas_training.calibrate datasets/crops_v5e --classifier runs/cls/crops_v5e/weights/best.pt

Classifier probabilities are over-confident in general; one temperature T, fitted on the
val crops by minimising the negative log-likelihood, makes p(ambulance) closer to the
real frequency (temperature scaling; it keeps the ranking, so it changes no ranking metric,
but it changes what a threshold like 0.5 means). Reports expected calibration error (ECE)
before and after, and per-crop precision/recall on val at the thresholds considered for
the track decision, broken down by crop width (the source crop size, from crops.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return 1 / (1 + np.exp(-logit(p) / t))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Grid search over T in [0.25, 10] (log scale), then a finer grid around the best."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(10), 200))
    best = min(grid, key=lambda t: nll(apply_temperature(p, t), y))
    fine = np.linspace(best * 0.9, best * 1.1, 50)
    return float(min(fine, key=lambda t: nll(apply_temperature(p, t), y)))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("crops", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from ultralytics import YOLO

    from .evaluate import class_index

    model = YOLO(str(args.classifier))
    amb = class_index(model.names, "ambulance")
    widths = {}
    meta = args.crops / "crops.csv"
    if meta.exists():
        for r in csv.DictReader(meta.open(encoding="utf-8")):
            widths[r["crop"]] = float(r["x2"]) - float(r["x1"])
    files, labels = [], []
    for label in ("ambulance", "other"):
        for f in sorted((args.crops / args.split / label).glob("*.jpg")):
            files.append(f)
            labels.append(1 if label == "ambulance" else 0)
    y = np.asarray(labels)
    p = np.asarray([r.probs.data[amb].item() for i in range(0, len(files), 64)
                    for r in model.predict([str(f) for f in files[i:i + 64]], verbose=False)])
    t = fit_temperature(p, y)
    pc = apply_temperature(p, t)
    report = {"split": args.split, "n": int(len(y)), "positives": int(y.sum()), "temperature": round(t, 3),
              "ece_before": round(ece(p, y), 4), "ece_after": round(ece(pc, y), 4), "thresholds": {}}
    w = np.asarray([widths.get(f.name, np.nan) for f in files])
    bins = [(0, 40), (40, 64), (64, 100), (100, 1e9)]
    for th in args.thresholds:
        row = {}
        for name, prob in (("raw", p), ("calibrated", pc)):
            pred = prob >= th
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            row[name] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
                         "recall": round(tp / (tp + fn), 3) if tp + fn else None}
        by_width = {}
        for lo, hi in bins:
            m = (w >= lo) & (w < hi) & (y == 1)
            if m.any():
                by_width[f"{int(lo)}-{int(hi) if hi < 1e9 else 'inf'}"] = {"n": int(m.sum()), "recall_raw": round(float((p[m] >= th).mean()), 3)}
        row["ambulance_recall_by_width"] = by_width
        report["thresholds"][str(th)] = row
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

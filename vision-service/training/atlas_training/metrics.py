"""Detection metrics for one class, independent of the model's own class list.

Pre-trained models have other classes (SmartTrafficAI: ambulance, car, fire_truck,
police) than our fine-tuned model (ambulance only), so Ultralytics' built-in validation
cannot compare them on one test set. Here every model is reduced to its ambulance boxes
and scored the same way:

- A prediction is a true positive if its IoU with a still unmatched ground-truth box is
  at least `iou`; predictions are matched in order of confidence (as in COCO / VOC).
- Precision and recall are reported at one confidence threshold (NFR-04 target: both
  at least 0.90).
- AP@0.5 is the area under the precision-recall curve with all-point interpolation.

Boxes are (x1, y1, x2, y2) in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageResult:
    """Ground truth and predictions of one image for the evaluated class."""

    gt: np.ndarray        # (n, 4)
    boxes: np.ndarray     # (m, 4)
    scores: np.ndarray    # (m,)


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float
    ap50: float
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_images: int
    conf: float

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a proportion; better than the normal approximation
    for the small counts of our clip-level tests (e.g. 7 of 8 videos)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def false_alarm_upper_bound(alarms: int, hours: float) -> float:
    """95 % upper bound on false alarms per hour (Poisson). For zero alarms this is the
    'rule of three': 3 / hours."""
    from math import exp, factorial

    if hours <= 0:
        return float("inf")
    # Smallest rate m with P(X <= alarms | m) <= 0.05, by bisection.
    lo, hi = 0.0, 10.0 * (alarms + 3)
    for _ in range(100):
        m = (lo + hi) / 2
        cdf = sum(exp(-m) * m**k / factorial(k) for k in range(alarms + 1))
        lo, hi = (m, hi) if cdf > 0.05 else (lo, m)
    return hi / hours


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for boxes in (x1, y1, x2, y2) form."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match(image: ImageResult, iou: float) -> np.ndarray:
    """True-positive flag for each prediction of one image (greedy, by confidence)."""
    order = np.argsort(-image.scores, kind="stable")
    ious = box_iou(image.boxes, image.gt)
    used = np.zeros(len(image.gt), dtype=bool)
    tp = np.zeros(len(image.boxes), dtype=bool)
    for i in order:
        if len(image.gt) == 0:
            break
        candidates = np.where(~used, ious[i], -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] >= iou:
            used[j] = True
            tp[i] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, all-point interpolation."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def score(images: list[ImageResult], iou: float = 0.5, conf: float = 0.25) -> Scores:
    """Precision, recall and F1 at `conf`, and AP@`iou` over all images."""
    flags, confs = [], []
    n_gt = 0
    for image in images:
        n_gt += len(image.gt)
        flags.append(match(image, iou))
        confs.append(np.asarray(image.scores, dtype=float))
    tp_flags = np.concatenate(flags) if flags else np.zeros(0, dtype=bool)
    all_conf = np.concatenate(confs) if confs else np.zeros(0)

    order = np.argsort(-all_conf, kind="stable")
    tp_cum = np.cumsum(tp_flags[order])
    fp_cum = np.cumsum(~tp_flags[order])
    if n_gt and len(order):
        ap = average_precision(tp_cum / n_gt, tp_cum / np.maximum(tp_cum + fp_cum, 1))
    else:
        ap = 0.0

    kept = all_conf >= conf
    tp = int(np.sum(tp_flags & kept))
    fp = int(np.sum(~tp_flags & kept))
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(precision, recall, f1, ap, tp, fp, fn, n_gt, len(images), conf)


def best_f1_threshold(images: list[ImageResult], iou: float = 0.5) -> Scores:
    """Scores at the confidence threshold with the highest F1 (reported alongside)."""
    thresholds = sorted({float(s) for image in images for s in image.scores})
    best = score(images, iou, 1.01)
    for t in thresholds:
        s = score(images, iou, t)
        if s.f1 > best.f1:
            best = s
    return best

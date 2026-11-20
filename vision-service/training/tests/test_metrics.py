import numpy as np
import pytest

from atlas_training.metrics import (
    ImageResult,
    average_precision,
    best_f1_threshold,
    box_iou,
    match,
    score,
)


def image(gt, boxes=(), scores=()):
    return ImageResult(
        gt=np.asarray(gt, dtype=float).reshape(-1, 4),
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        scores=np.asarray(scores, dtype=float),
    )


def test_box_iou():
    a = [[0, 0, 10, 10]]
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)
    assert box_iou(a, [[20, 20, 30, 30]])[0, 0] == 0
    assert box_iou(a, np.zeros((0, 4))).shape == (1, 0)


def test_each_ground_truth_box_is_matched_once():
    # Two predictions on the same ambulance: the more confident one is the hit.
    img = image(gt=[[0, 0, 10, 10]], boxes=[[0, 0, 10, 10], [1, 1, 10, 10]], scores=[0.6, 0.9])
    assert match(img, 0.5).tolist() == [False, True]


def test_low_overlap_is_a_false_positive():
    img = image(gt=[[0, 0, 10, 10]], boxes=[[6, 0, 16, 10]], scores=[0.9])
    assert match(img, 0.5).tolist() == [False]


def test_perfect_detector():
    s = score([image([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.9])], iou=0.5, conf=0.25)
    assert (s.precision, s.recall, s.ap50, s.tp, s.fp, s.fn) == (1.0, 1.0, 1.0, 1, 0, 0)


def test_threshold_and_background_images():
    images = [
        image([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.8]),     # hit
        image([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.1]),     # hit, but below conf
        image([], [[0, 0, 10, 10]], [0.7]),                    # false alarm on a background image
    ]
    s = score(images, iou=0.5, conf=0.25)
    assert (s.tp, s.fp, s.fn, s.n_gt, s.n_images) == (1, 1, 1, 2, 3)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    # Ranked: 0.8 TP, 0.7 FP, 0.1 TP -> AP = 0.5*1 + 0.5*(2/3)
    assert s.ap50 == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_no_predictions():
    s = score([image([[0, 0, 10, 10]])])
    assert (s.precision, s.recall, s.ap50, s.fn) == (0.0, 0.0, 0.0, 1)


def test_average_precision_of_step_curve():
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 0.5])) == pytest.approx(0.75)


def test_best_f1_threshold_drops_low_confidence_false_alarms():
    images = [
        image([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.9]),
        image([], [[0, 0, 10, 10]], [0.3]),
    ]
    best = best_f1_threshold(images)
    assert best.conf == pytest.approx(0.9)
    assert (best.precision, best.recall) == (1.0, 1.0)

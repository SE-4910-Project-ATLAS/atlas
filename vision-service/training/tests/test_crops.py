from pathlib import Path

import numpy as np

from atlas_training.crops import negatives, sample_negatives, square_crop


def test_square_crop_pads_and_stays_square_at_the_edge():
    image = np.zeros((100, 200, 3), np.uint8)
    crop = square_crop(image, (0, 0, 40, 20), pad=0.25)
    assert crop.shape == (60, 60, 3)


def test_negatives_skip_vehicles_on_an_ambulance():
    vehicles = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    ambulances = np.array([[0, 0, 10, 11]], float)
    assert negatives(vehicles, np.ones(2), ambulances) == [1]
    assert negatives(vehicles, np.ones(2), np.zeros((0, 4))) == [0, 1]


def test_sample_negatives_keeps_all_own_frames_first():
    items = [(Path(f"own_cctv__a{i}.jpg"),) for i in range(3)] + [(Path(f"special__b{i}.jpg"),) for i in range(10)]
    out = sample_negatives(items, 5, np.random.default_rng(0))
    assert len(out) == 5
    assert sum(x[0].name.startswith("own_cctv__") for x in out) == 3


def test_fuse_keeps_one_stage_and_adds_only_confident_new_boxes():
    from atlas_training.metrics import ImageResult
    from atlas_training.twostage import fuse

    gt = np.zeros((0, 4))
    one = ImageResult(gt=gt, boxes=np.array([[0, 0, 10, 10]], float), scores=np.array([0.3]))
    two = ImageResult(gt=gt, boxes=np.array([[0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]], float),
                      scores=np.array([0.9, 0.97, 0.5]))
    f = fuse(one, two, min_two=0.95)
    assert f.boxes.tolist() == [[0, 0, 10, 10], [50, 50, 60, 60]]
    assert f.scores.tolist() == [0.9, 0.97]


def test_square_crop_matches_padding_the_whole_frame():
    import cv2
    import numpy as np

    from atlas_training.crops import square_crop

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)

    def reference(box, pad):
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
        a, b = int(round((x1 + x2) / 2 - side / 2)), int(round((y1 + y2) / 2 - side / 2))
        s = max(1, int(round(side)))
        t, l = max(0, -b), max(0, -a)
        p = cv2.copyMakeBorder(image, t, max(0, b + s - 120), l, max(0, a + s - 200), cv2.BORDER_REPLICATE)
        return p[b + t:b + t + s, a + l:a + l + s]

    for box in [(50, 30, 90, 70), (-10, -5, 30, 40), (170, 90, 230, 130), (0, 0, 200, 120), (5, 50, 15, 52)]:
        assert np.array_equal(square_crop(image, box, 0.2), reference(box, 0.2)), box

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

import numpy as np

from atlas_training.propagate import ambulance_tracks, to_xyxy


def test_to_xyxy():
    np.testing.assert_allclose(to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]), 100, 50), [[40, 15, 60, 35]])


def test_track_matching_a_labelled_ambulance_is_found():
    labels = {10: np.array([[0.5, 0.5, 0.2, 0.4]]), 20: np.zeros((0, 4))}
    frames = [(10, {1: np.array([40, 15, 60, 35.0]), 2: np.array([0, 0, 10, 10.0])}),
              (20, {1: np.array([40, 15, 60, 35.0])})]
    assert ambulance_tracks(frames, labels, (100, 50), 0.5) == {1: [10]}

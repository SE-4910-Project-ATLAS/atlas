import numpy as np

from atlas_training import frames


def test_pick_evenly_keeps_first_and_last():
    assert frames.pick_evenly(list(range(10)), 4) == [0, 3, 6, 9]
    assert frames.pick_evenly([1, 2], 5) == [1, 2]
    assert frames.pick_evenly([7, 8, 9], 1) == [7]


def test_keep_distinct_drops_repeats_of_the_last_kept_frame():
    rng = np.random.default_rng(0)
    a = frames.thumbnail(rng.integers(0, 255, (64, 64), dtype=np.uint8))
    b = frames.thumbnail(rng.integers(0, 255, (64, 64), dtype=np.uint8))
    # a, a, b, b, a: the repeats are dropped, a change back to a is kept
    assert frames.keep_distinct([a, a, b, b, a], 0.97) == [0, 2, 4]

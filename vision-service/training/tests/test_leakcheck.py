import numpy as np

from atlas_training.leakcheck import matching_frames


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_matching_frames_counts_frames_with_a_close_frame_in_the_other_video():
    a = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = _unit([[1, 0.05, 0], [0.7, 0.7, 0]])
    assert matching_frames(a, b, 0.99) == 1       # only the first frame has a near copy
    assert matching_frames(a, b, 0.7) == 2


def test_matching_frames_with_an_empty_video_is_zero():
    assert matching_frames(np.zeros((0, 3)), _unit([[1, 0, 0]]), 0.5) == 0


def test_boilerplate_marks_frames_shared_by_three_videos_only():
    from atlas_training.leakcheck import boilerplate

    card = [0, 0, 1]
    videos = {"a": _unit([[1, 0, 0], card]), "b": _unit([[0, 1, 0], card]), "c": _unit([[1, 1, 0], card]),
              "d": _unit([[1, 0, 0]])}                     # d shares real footage with a only
    m = boilerplate(videos, 0.99)
    assert m["a"].tolist() == [False, True] and m["c"].tolist() == [False, True]
    assert m["d"].tolist() == [False]

import numpy as np

from atlas_training.mine import Detections, Example, find_flickers, find_gaps, thin_out


def frame(*boxes, score=0.9):
    if not boxes:
        return Detections(np.zeros((0, 4)), np.zeros(0))
    return Detections(np.array(boxes, dtype=float), np.full(len(boxes), score))


AMB = (100, 100, 200, 160)
VAN = (500, 300, 560, 340)


def test_steady_detection_is_neither_gap_nor_flicker():
    frames = [frame(AMB) for _ in range(12)]
    assert find_gaps(frames, 5, 0.5, 0.2) == []
    assert find_flickers(frames, 5, 0.5, 0.2) == []


def test_missing_frame_between_detections_is_a_gap_with_interpolated_box():
    frames = [frame((100 + 10 * i, 100, 200 + 10 * i, 160)) for i in range(9)]
    frames[4] = frame()
    gaps = find_gaps(frames, 5, 0.5, 0.2)
    assert [g.index for g in gaps] == [4]
    np.testing.assert_allclose(gaps[0].box, [140, 100, 240, 160])


def test_no_gap_when_the_object_has_left():
    frames = [frame(AMB)] * 4 + [frame()] * 8
    assert find_gaps(frames, 5, 0.5, 0.2) == []


def test_weak_neighbours_do_not_make_a_gap():
    frames = [frame(AMB, score=0.3)] * 4 + [frame()] + [frame(AMB, score=0.3)] * 4
    assert find_gaps(frames, 5, 0.5, 0.2) == []


def test_isolated_confident_detection_is_a_flicker():
    frames = [frame(AMB) for _ in range(13)]
    frames[6] = frame(AMB, VAN)
    flickers = find_flickers(frames, 5, 0.5, 0.2)
    assert [(f.index, tuple(f.box)) for f in flickers] == [(6, VAN)]


def test_thin_out_keeps_one_per_distance_and_kind():
    box = np.zeros(4)
    examples = [Example(i, "gap", box) for i in range(0, 30, 2)] + [Example(5, "flicker", box)]
    kept = thin_out(examples, 12, 10)
    assert [(e.index, e.kind) for e in kept] == [(0, "gap"), (5, "flicker"), (12, "gap"), (24, "gap")]
    assert len(thin_out(examples, 1, 4)) == 4

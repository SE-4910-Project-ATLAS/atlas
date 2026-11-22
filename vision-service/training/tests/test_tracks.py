from atlas_training.tracks import TrackVotes, alarm_from_tracks


def test_track_mean_needs_enough_frames_and_mean():
    v = TrackVotes()
    for p in (0.9, 0.9):
        v.add(1, p)
    assert not v.is_ambulance(1, min_frames=3, threshold=0.5)
    v.add(1, 0.0)
    assert v.is_ambulance(1, min_frames=3, threshold=0.5)
    assert not v.is_ambulance(1, min_frames=3, threshold=0.7)


def test_one_confident_frame_does_not_raise_an_alarm():
    frames = [{1: 0.1}, {1: 0.99}, {1: 0.1}, {1: 0.1}]
    assert alarm_from_tracks(frames, min_frames=3, threshold=0.5) == [False] * 4


def test_alarm_follows_the_track_while_it_is_visible():
    frames = [{1: 0.8}, {1: 0.8}, {1: 0.8, 2: 0.1}, {2: 0.1}, {1: 0.8}]
    assert alarm_from_tracks(frames, min_frames=3, threshold=0.5) == [False, False, True, False, True]

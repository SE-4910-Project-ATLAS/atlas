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


def test_sprt_needs_several_confident_frames():
    from atlas_training.tracks import alarm_sprt

    frames = [{1: 0.99}, {1: 0.99}, {1: 0.99}]
    assert alarm_sprt(frames, upper=5.0, clip=2.0) == [False, False, True]
    assert alarm_sprt([{1: 0.99}, {1: 0.01}, {1: 0.99}, {1: 0.99}], upper=5.0) == [False] * 4


def test_time_rule_does_not_depend_on_read_rate():
    from atlas_training.tracks import alarm_from_tracks_time

    slow = alarm_from_tracks_time([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [{1: 0.9}] * 6, 1.0, 0.5)
    fast = alarm_from_tracks_time([i * 0.08 for i in range(14)], [{1: 0.9}] * 14, 1.0, 0.5)
    assert slow[-1] and not any(slow[:-1])
    assert fast[-1] and not any(fast[:-1])


def test_only_moving_drops_parked_tracks():
    from atlas_training.tracks import only_moving

    frames = [{1: 0.9, 2: 0.9}] * 3
    boxes = [{"1": [100, 100, 50], "2": [300, 300, 50]},
             {"1": [100, 101, 50], "2": [330, 300, 50]},
             {"1": [101, 100, 50], "2": [360, 300, 50]}]
    out = only_moving(frames, boxes, 1.0)
    assert out == [{}, {}, {2: 0.9}]


def test_a_parked_vehicle_found_again_alarms_once():
    from atlas_training.tracks import alarm_once_per_place

    # Track 1 meets the rule at (100, 100), is lost; track 2 is found at the same place.
    times = [i * 0.2 for i in range(20)]
    frames = [{1: 0.9}] * 8 + [{}] * 2 + [{2: 0.9}] * 10
    boxes = [{1: [100, 100, 50]}] * 8 + [{}] * 2 + [{2: [102, 101, 50]}] * 10
    alarm = alarm_once_per_place(times, frames, boxes, min_frames=5, threshold=0.5)
    assert any(alarm[:8])
    assert not any(alarm[10:])


def test_a_track_held_back_at_an_old_place_is_released_when_it_moves():
    from atlas_training.tracks import alarm_once_per_place

    times = [i * 0.2 for i in range(20)]
    frames = [{1: 0.9}] * 6 + [{2: 0.9}] * 14
    boxes = [{1: [100, 100, 50]}] * 6 + [{2: [100 + 2 * i, 100, 50]} for i in range(14)]
    alarm = alarm_once_per_place(times, frames, boxes, min_frames=5, threshold=0.5)
    assert alarm[5]        # track 1 alarms at a new place
    assert not alarm[11]   # track 2 meets the rule 10 px from that place: held back
    assert alarm[19]       # 26 px away (more than half a box width): released


def test_a_short_confident_track_alarms_only_with_the_short_rule():
    from atlas_training.tracks import alarm_once_per_place

    times = [i * 0.2 for i in range(4)]
    frames = [{1: 0.99}] * 3 + [{}]
    boxes = [{1: [100, 100, 50]}] * 3 + [{}]
    assert not any(alarm_once_per_place(times, frames, boxes, min_frames=5, threshold=0.7))
    assert alarm_once_per_place(times, frames, boxes, 5, 0.7, short_frames=3, short_threshold=0.9)[2]
    weak = [{1: 0.8}] * 3 + [{}]          # confident enough for the long rule, not the short one
    assert not any(alarm_once_per_place(times, weak, boxes, 5, 0.7, short_frames=3, short_threshold=0.9))

from atlas_training.events import Event, alarm_spans, events_from_labels, score_video, smooth


def test_smooth_ignores_a_single_frame_and_follows_a_steady_detection():
    hits = [False] * 5 + [True] + [False] * 5 + [True] * 6
    alarm = smooth(hits, buffer=6, min_frames=3, ratio=0.5)
    assert not any(alarm[:11])
    assert alarm[-1]


def test_smooth_keeps_alarm_through_a_short_occlusion():
    hits = [True] * 6 + [False, False] + [True] * 3
    alarm = smooth(hits, buffer=6, min_frames=3, ratio=0.5)
    assert all(alarm[2:])


def test_events_split_on_empty_label_and_long_gap():
    events = events_from_labels([1.0, 1.5, 2.0, 10.0, 10.5, 20.0], [3.0], merge=3.0)
    assert [(e.start, e.end) for e in events] == [(1.0, 2.0), (10.0, 10.5), (20.0, 20.0)]


def test_alarm_spans():
    spans = alarm_spans([0, 1, 2, 3, 4, 5], [False, True, True, False, True, False])
    assert [(s.start, s.end) for s in spans] == [(1, 2), (4, 4)]


def test_score_video_counts_found_events_delay_and_false_alarms():
    events = [Event(2.0, 4.0), Event(10.0, 11.0)]
    times = [float(t) for t in range(15)]
    alarm = [t in (3, 4) or t == 7 for t in range(15)]
    r = score_video(events, times, alarm, margin=1.0)
    assert r["events"] == 2 and r["found"] == 1
    assert r["delays_s"] == [1.0]
    assert r["false_alarms"] == 1
    assert r["duration_s"] == 15.0


def test_split_at_cuts_merges_within_a_shot_and_never_across():
    from atlas_training.events import split_at_cuts

    events = [Event(1, 1), Event(5, 6), Event(12, 13), Event(14, 14)]
    out = split_at_cuts(events, cuts=[10.0])
    assert [(e.start, e.end) for e in out] == [(1, 6), (12, 14)]

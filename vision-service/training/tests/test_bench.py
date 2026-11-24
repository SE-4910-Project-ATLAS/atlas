import numpy as np

from atlas_training.bench import bootstrap, load_tracks, rates, track_raised


def _tracks():
    return [{"id": "a1", "label": "ambulance", "files": [0] * 6},
            {"id": "a2", "label": "ambulance", "files": [0] * 6},
            {"id": "o1", "label": "other", "files": [0] * 6},
            {"id": "o2", "label": "other", "files": [0] * 6}]


def test_track_raised_needs_frames_and_mean():
    assert track_raised(np.array([0.9] * 5), 5, 0.7)
    assert not track_raised(np.array([0.9] * 4), 5, 0.7)
    assert not track_raised(np.array([0.9, 0.9, 0.9, 0.2, 0.2]), 5, 0.7)


def test_rates_split_by_label():
    probs = {"a1": np.full(6, 0.9), "a2": np.full(6, 0.1), "o1": np.full(6, 0.8), "o2": np.full(6, 0.1)}
    r = rates(_tracks(), probs, np.arange(4), 5, 0.7)
    assert r["recall_tracks"] == 0.5 and r["false_alarm_tracks"] == 0.5
    assert r["recall_crops"] == 0.5 and r["false_alarm_crops"] == 0.5


def test_paired_difference_is_clear_only_when_it_holds_everywhere():
    tracks = [{"id": f"o{i}", "label": "other", "files": [0] * 5} for i in range(40)]
    tracks += [{"id": f"a{i}", "label": "ambulance", "files": [0] * 5} for i in range(10)]
    bad = {t["id"]: np.full(5, 0.9) for t in tracks}                       # raises every track
    good = {t["id"]: np.full(5, 0.9 if t["label"] == "ambulance" else 0.1) for t in tracks}
    out = bootstrap(tracks, {"bad": bad, "good": good, "bad2": dict(bad)}, 5, 0.7, n=200)
    assert out["pairs"]["good - bad"]["false_alarm_tracks"]["clear"]
    assert out["pairs"]["good - bad"]["false_alarm_tracks"]["diff"] == -1.0
    assert not out["pairs"]["bad2 - bad"]["false_alarm_tracks"]["clear"]


def test_load_tracks_reads_verdicts_and_time_range(tmp_path):
    root = tmp_path / "mine"
    (root / "vid" / "7").mkdir(parents=True)
    for t in ("00001.00_0.900", "00005.00_0.800", "00009.00_0.100"):
        (root / "vid" / "7" / f"{t}.jpg").write_bytes(b"x")
    (root / "verdicts.csv").write_text("video,track,verdict,from_s,until_s,note\nvid,7,ambulance,0,6,\nvid,8,skip,,,\n",
                                       encoding="utf-8")
    tracks = load_tracks([root])
    assert len(tracks) == 1 and tracks[0]["label"] == "ambulance" and len(tracks[0]["files"]) == 2


def test_load_tracks_keeps_at_most_max_crops_spread_over_the_track(tmp_path):
    root = tmp_path / "mine"
    (root / "v" / "1").mkdir(parents=True)
    for i in range(100):
        (root / "v" / "1" / f"{i:08.2f}_0.100.jpg").write_bytes(b"x")
    (root / "verdicts.csv").write_text("video,track,verdict,from_s,until_s,note\nv,1,other,,,\n", encoding="utf-8")
    files = load_tracks([root], max_crops=10)[0]["files"]
    assert len(files) == 10 and files[0].name.startswith("00000.00") and files[-1].name.startswith("00099.00")

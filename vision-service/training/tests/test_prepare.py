from pathlib import Path

import pytest
import yaml

from atlas_training import config as cfg
from atlas_training import prepare

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "ambulance_v1.yaml"


def test_example_config_loads():
    exp = cfg.load(CONFIG)
    assert exp.target_classes == ["ambulance"]
    assert {d.id for d in exp.datasets} == {"emergency_vehicles_detection", "special_vehicles_small_ds"}


def test_config_rejects_unknown_class_map_value(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["datasets"][0]["class_map"]["ambulance"] = "fire_truck"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="class_map"):
        cfg.load(bad)


def test_parse_box_and_polygon():
    assert prepare.parse_label_line("0 0.5 0.5 0.2 0.4") == (0, 0.5, 0.5, 0.2, 0.4)
    cls, cx, cy, w, h = prepare.parse_label_line("1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
    assert (cls, cx, cy, w, h) == pytest.approx((1, 0.2, 0.4, 0.2, 0.4))
    assert prepare.parse_label_line("   ") is None


def test_map_labels_keeps_ignores_and_drops():
    names = ["Ambulance", "Police car", "emergency"]
    class_map = {"Ambulance": "ambulance", "Police car": cfg.IGNORE, "emergency": cfg.DROP_IMAGE}
    report = prepare.SourceReport()

    kept = prepare.map_labels([(0, .5, .5, .1, .1), (1, .2, .2, .1, .1)], names, class_map, ["ambulance"], report)
    assert kept == [(0, .5, .5, .1, .1)]
    assert report.ignored_boxes["Police car"] == 1

    # The ignored box in a dropped image is not counted as ignored.
    dropped = prepare.map_labels([(1, .5, .5, .1, .1), (2, .2, .2, .1, .1)], names, class_map, ["ambulance"], report)
    assert dropped is None
    assert report.dropped_images["emergency"] == 1
    assert report.ignored_boxes["Police car"] == 1


def samples(n, origin):
    return [prepare.Sample("s", Path(f"{origin}{i}.jpg"), [], origin) for i in range(n)]


def test_source_test_split_is_kept():
    # collect() stores Roboflow's "valid" folder as origin_split "val"
    out = prepare.assign_splits(samples(80, "train") + samples(10, "val") + samples(10, "test"), 0.15, 0.15, 1)
    assert [len(out[k]) for k in ("train", "val", "test")] == [80, 10, 10]
    assert all(s.origin_split == "test" for s in out["test"])


def test_missing_splits_are_sampled_with_the_seed():
    first = prepare.assign_splits(samples(100, "train"), 0.15, 0.15, 7)
    again = prepare.assign_splits(samples(100, "train"), 0.15, 0.15, 7)
    assert [len(first[k]) for k in ("train", "val", "test")] == [70, 15, 15]
    assert [s.image for s in first["test"]] == [s.image for s in again["test"]]
    assert not {s.image for s in first["test"]} & {s.image for s in first["train"]}


def copies(n_groups, per_group, origin, labelled=True):
    box = [(0, .5, .5, .1, .1)] if labelled else []
    return [
        prepare.Sample("s", Path(f"{origin}{g}_jpg.rf.{c}.jpg"), box, origin, f"{origin}{g}_jpg")
        for g in range(n_groups) for c in range(per_group)
    ]


def splits_of_groups(out):
    seen = {}
    for split, items in out.items():
        for s in items:
            seen.setdefault(s.group, set()).add(split)
    return seen


def test_sampled_splits_never_separate_copies():
    out = prepare.assign_splits(copies(40, 3, "train"), 0.15, 0.15, 3)
    assert all(len(splits) == 1 for splits in splits_of_groups(out).values())
    assert out["test"] and out["val"]


def test_move_to_test_moves_whole_labelled_groups_only():
    samples = copies(50, 3, "train") + copies(30, 1, "bg", labelled=False) + copies(5, 1, "test", labelled=False)
    for s in samples[-5:]:
        s.origin_split = "test"
    out = prepare.assign_splits(samples, 0.15, 0.15, 3, move_to_test=0.2)
    moved = [s for s in out["test"] if s.origin_split != "test"]
    assert len(moved) == 10 * 3                       # 20% of 50 groups, with all copies
    assert all(s.labels for s in moved)               # background images stay where they were
    assert all(len(splits) == 1 for splits in splits_of_groups(out).values())


def test_group_key_uses_the_original_name_of_shortened_files():
    original = "very_long_name_" + "x" * 90 + "_jpg.rf.abc123"
    short = cfg.short_stem(original)
    assert prepare.group_key(short, {short: original}) == "very_long_name_" + "x" * 90 + "_jpg"
    assert prepare.group_key("100_jpg.rf.7924ca", {}) == "100_jpg"


def test_clip_with_a_frame_in_the_source_test_split_goes_to_test_whole():
    samples = copies(10, 1, "train") + copies(2, 1, "val") + copies(1, 1, "test")
    samples[-1].origin_split = "test"
    for s in samples[:3] + samples[-1:]:
        s.clip = "clipA"                       # three train frames share a clip with the test frame
    out = prepare.assign_splits(samples, 0.15, 0.15, 1)
    assert {s.image for s in samples[:3]} <= {s.image for s in out["test"]}
    assert len(out["train"]) == 7


def test_move_to_test_moves_whole_clips():
    samples = copies(30, 3, "train")
    for i, s in enumerate(samples):
        s.clip = f"clip{i // 9}"               # 10 clips of 3 images x 3 copies
    out = prepare.assign_splits(samples, 0.15, 0.15, 5, move_to_test=0.2)
    clips_by_split = {}
    for split, items in out.items():
        for s in items:
            clips_by_split.setdefault(s.clip, set()).add(split)
    assert all(len(v) == 1 for v in clips_by_split.values())
    assert sum("test" in v for v in clips_by_split.values()) >= 2


def test_assign_clips_links_rotated_copies_and_similar_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    scene_a, scene_b = rng.integers(0, 255, (64, 64), np.uint8), rng.integers(0, 255, (64, 64), np.uint8)
    frames = {
        "a1": scene_a,
        "a1_rotated": np.rot90(scene_a, 3),
        "a2": np.clip(scene_a.astype(int) + 5, 0, 255).astype(np.uint8),   # next frame, same camera
        "b1": scene_b,
    }
    samples = []
    for name, image in frames.items():
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.ascontiguousarray(image))
        samples.append(prepare.Sample("s", path, [], "train"))
    assert prepare.assign_clips(samples, 0.85) == 2
    clip = {s.image.stem: s.clip for s in samples}
    assert clip["a1"] == clip["a1_rotated"] == clip["a2"] != clip["b1"]


def test_orientation_override_replaces_the_detector(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare, "best_orientation",
                        lambda *a, **kw: calls.append(a) or ((1.0, a[0][0], 0) if kw.get("with_score") else (a[0][0], 0)))
    import sys as _sys
    import types
    monkeypatch.setitem(_sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: None))
    pytest.importorskip("cv2")
    group = [prepare.Sample("s", Path(f"img_jpg.rf.{c}.jpg"), [(0, .7, .2, .4, .2)], "train", "img_jpg") for c in "abc"]
    other = [prepare.Sample("s", Path("x_jpg.rf.a.jpg"), [(0, .5, .5, .1, .1)], "train", "x_jpg")]
    kept, stats = prepare.upright_test_copies(group + other, tmp_path / "w.pt", {"img_jpg.rf.b": 1})
    chosen = [s for s in kept if s.group == "img_jpg"]
    assert [s.image.stem for s in chosen] == ["img_jpg.rf.b"]
    assert chosen[0].rotate == 1 and chosen[0].labels == [prepare.rotate_label((0, .7, .2, .4, .2), 1)]
    assert stats["set_by_hand"] == 1 and len(calls) == 1   # only the other image used the detector


def test_frames_of_a_clip_are_aligned_to_the_reference(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import sys as _sys
    import types
    monkeypatch.setitem(_sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: None))
    scene = np.random.default_rng(1).integers(0, 255, (64, 64), np.uint8)
    samples = []
    # frame 1: one copy, already upright; frame 2: two copies, turned 1 and 2 quarter turns
    for name, group, k in [("f1_jpg.rf.a", "f1_jpg", 0), ("f2_jpg.rf.a", "f2_jpg", 1), ("f2_jpg.rf.b", "f2_jpg", 2)]:
        path = tmp_path / f"{name}.png"
        frame = np.clip(scene.astype(int) + (3 if group == "f2_jpg" else 0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), np.ascontiguousarray(np.rot90(frame, k)))
        samples.append(prepare.Sample("s", path, [(0, .5, .5, .1, .1)], "train", group, clip="clip1"))
    # The detector trusts frame 1 upright (score 9) more than frame 2 (score 1).
    monkeypatch.setattr(prepare, "best_orientation",
                        lambda copies_, *a, **kw: (9.0 if copies_[0].group == "f1_jpg" else 1.0, copies_[0], 0))
    kept, stats = prepare.upright_test_copies(samples, tmp_path / "w.pt")
    f2 = next(s for s in kept if s.group == "f2_jpg")
    # rotating the copy back by (4 - k) quarter turns makes it upright again
    assert (4 - {"f2_jpg.rf.a": 1, "f2_jpg.rf.b": 2}[f2.image.stem]) % 4 == f2.rotate
    assert stats["aligned_to_clip"] == 1


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_rotated_label_follows_the_rotated_image(quarter_turns):
    np = pytest.importorskip("numpy")
    h, w = 60, 100
    image = np.zeros((h, w), dtype=np.uint8)
    image[10:30, 50:90] = 1                       # box x 50..90, y 10..30
    box = (0, 70 / w, 20 / h, 40 / w, 20 / h)
    rotated = np.rot90(image, quarter_turns)
    ys, xs = np.nonzero(rotated)
    rh, rw = rotated.shape
    expected = ((xs.min() + xs.max() + 1) / 2 / rw, (ys.min() + ys.max() + 1) / 2 / rh,
                (xs.max() - xs.min() + 1) / rw, (ys.max() - ys.min() + 1) / rh)
    _, cx, cy, bw, bh = prepare.rotate_label(box, quarter_turns)
    assert (cx, cy, bw, bh) == pytest.approx(expected)


def test_unlisted_label_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DOWNLOAD_DIR", tmp_path)
    source = cfg.DatasetSource("demo", "w", "p", 1, "CC BY 4.0", "u", {"ambulance": "ambulance"})
    source.download_dir.mkdir(parents=True)
    (source.download_dir / "data.yaml").write_text(yaml.safe_dump({"names": ["ambulance", "bus"]}))
    with pytest.raises(SystemExit, match="bus"):
        prepare.collect(source, ["ambulance"], prepare.SourceReport())


@pytest.mark.parametrize("augmented", [True, False])
def test_copies_are_grouped_by_name_only_for_augmented_exports(tmp_path, monkeypatch, augmented):
    monkeypatch.setattr(cfg, "DOWNLOAD_DIR", tmp_path)
    source = cfg.DatasetSource("demo", "w", "p", 1, "CC BY 4.0", "u", {"ambulance": "ambulance"},
                               augmented_copies=augmented)
    root = source.download_dir
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["ambulance"]}))
    for name in ("0053_jpg.rf.aaa", "0053_jpg.rf.bbb"):
        (root / "train" / "images" / f"{name}.jpg").write_bytes(b"")
    samples = prepare.collect(source, ["ambulance"], prepare.SourceReport())
    keys = {prepare.copy_key(s) for s in samples}
    assert len(keys) == (1 if augmented else 2)


def exclude_source(tmp_path, monkeypatch, exclude):
    monkeypatch.setattr(cfg, "DOWNLOAD_DIR", tmp_path)
    source = cfg.DatasetSource("demo", "w", "p", 1, "CC BY 4.0", "u", {"ambulance": "ambulance"},
                               exclude_images=exclude)
    root = source.download_dir
    (root / "train" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["ambulance"]}))
    for name in ("good", "bad"):
        (root / "train" / "images" / f"{name}.jpg").write_bytes(b"")
    return source


def test_excluded_images_are_left_out_and_counted(tmp_path, monkeypatch):
    source = exclude_source(tmp_path, monkeypatch, {"bad": "box on the light bar only"})
    report = prepare.SourceReport()
    samples = prepare.collect(source, ["ambulance"], report)
    assert [s.image.stem for s in samples] == ["good"]
    assert report.excluded == 1


def test_excluded_image_that_does_not_exist_stops_the_run(tmp_path, monkeypatch):
    source = exclude_source(tmp_path, monkeypatch, {"typo": "reason"})
    with pytest.raises(SystemExit, match="typo"):
        prepare.collect(source, ["ambulance"], prepare.SourceReport())


def test_test_images_that_copy_a_training_image_are_dropped(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(4)
    seen, unseen = rng.integers(0, 255, (64, 64), np.uint8), rng.integers(0, 255, (64, 64), np.uint8)

    def sample(name, image, split, labels=()):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.ascontiguousarray(image))
        return prepare.Sample("a", path, list(labels), split)

    train = [sample("train_seen", seen, "train")]
    test = [
        sample("same_photo", seen, "test", [(0, .5, .5, .1, .1)]),
        sample("same_photo_rotated", np.rot90(seen), "test"),
        sample("new_scene", unseen, "test"),
    ]
    splits = {"a": {"train": train, "val": [], "test": test}}
    reports = {"a": prepare.SourceReport()}
    prepare.drop_test_near_copies(splits, 0.95, reports)
    assert [s.image.stem for s in splits["a"]["test"]] == ["new_scene"]
    assert reports["a"].near_copies_dropped == {"images": 2, "with_target": 1}
    assert splits["a"]["train"] == train

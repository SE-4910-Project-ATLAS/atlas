from atlas_training.addneg import chosen


def test_chosen_skips_excluded_and_low_scores(tmp_path):
    for name in ("0.950_amb.jpg", "0.400_van.jpg", "0.010_car.jpg"):
        (tmp_path / name).write_bytes(b"x")
    names = [f.name for f in chosen(tmp_path, {"0.950_amb.jpg"}, 0.0)]
    assert names == ["0.010_car.jpg", "0.400_van.jpg"]
    assert [f.name for f in chosen(tmp_path, set(), 0.3)] == ["0.400_van.jpg", "0.950_amb.jpg"]

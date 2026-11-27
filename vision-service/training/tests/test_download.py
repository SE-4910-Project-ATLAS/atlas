import zipfile

from atlas_training import config as cfg
from atlas_training.download import extract_short, shorten_in_place

LONG = "Screenshot-2025-04-28-at-18-09-27-" + "x" * 150 + ".rf.c19b6e868bff24183a5d2b68f42d936f"


def test_short_stem_keeps_short_names_and_is_stable():
    assert cfg.short_stem("img_001.rf.abc") == "img_001.rf.abc"
    short = cfg.short_stem(LONG)
    assert len(short) == 53
    assert short == cfg.short_stem(LONG)
    assert short != cfg.short_stem(LONG + "2")


def test_extract_short_keeps_image_and_label_together(tmp_path):
    archive = tmp_path / "roboflow.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data.yaml", "names: [Ambulance]\n")
        z.writestr(f"train/images/{LONG}.jpg", b"jpg")
        z.writestr(f"train/labels/{LONG}.txt", "0 0.5 0.5 0.1 0.1\n")
        z.writestr("valid/images/short.jpg", b"jpg")
    target = tmp_path / "out"
    stale = target / "train" / "images" / "half_written.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"")

    renamed = extract_short(archive, target)

    short = cfg.short_stem(LONG)
    assert (target / "train/images" / f"{short}.jpg").read_bytes() == b"jpg"
    assert (target / "train/labels" / f"{short}.txt").exists()
    assert (target / "valid/images/short.jpg").exists()
    assert (target / "data.yaml").exists()
    assert not stale.exists()
    assert len(renamed) == 2


def test_shorten_in_place(tmp_path):
    images = tmp_path / "test" / "images"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"")
    (images / "b" ).mkdir()
    long_name = "y" * 100 + ".jpg"
    (images / long_name).write_bytes(b"")
    renamed = shorten_in_place(tmp_path)
    assert renamed == [("test/images", long_name, cfg.short_stem("y" * 100) + ".jpg")]
    assert (images / "a.jpg").exists()

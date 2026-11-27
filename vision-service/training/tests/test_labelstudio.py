import pytest

from atlas_training import labelstudio


def test_prediction_converts_yolo_centre_boxes_to_percent_corners():
    pred = labelstudio.prediction(["0 0.5 0.4 0.2 0.1 0.83"], "v2")
    (box,) = pred["result"]
    assert box["value"]["x"] == pytest.approx(40)
    assert box["value"]["y"] == pytest.approx(35)
    assert box["value"]["width"] == pytest.approx(20)
    assert box["value"]["height"] == pytest.approx(10)
    assert box["score"] == pytest.approx(0.83)
    assert pred["model_version"] == "v2"


def test_annotation_round_trips_to_yolo():
    pred = labelstudio.prediction(["0 0.5 0.4 0.2 0.1"], "v2")
    pred["result"][0]["value"]["rectanglelabels"] = ["police"]
    (line,) = labelstudio.yolo_lines(pred)
    cls, *box = line.split()
    assert cls == "1"
    assert [float(v) for v in box] == pytest.approx([0.5, 0.4, 0.2, 0.1])


def test_clip_of_frame_name():
    assert labelstudio.clip_of("-lZjalQE3q4_f000125") == "-lZjalQE3q4"


def test_frame_without_boxes_has_no_prediction(tmp_path):
    images = tmp_path / "cctv" / "images"
    images.mkdir(parents=True)
    image = images / "clip_f000010.jpg"
    image.write_bytes(b"")
    (tmp_path / "cctv" / "labels").mkdir()
    (tmp_path / "cctv" / "labels" / "clip_f000010.txt").write_text("")
    item = labelstudio.task(image, tmp_path, tmp_path / "cctv" / "labels" / "clip_f000010.txt", "v2")
    assert item == {"data": {"image": "/data/local-files/?d=cctv/images/clip_f000010.jpg"}}

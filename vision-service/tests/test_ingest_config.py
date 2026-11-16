import pytest

from app.ingest import SourceType, load_cameras


def write(tmp_path, text):
    path = tmp_path / "cameras.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_all_source_types(tmp_path):
    path = write(
        tmp_path,
        """
[[camera]]
id = "a"
type = "mjpeg"
url = "http://example.invalid/a"

[[camera]]
id = "b"
type = "hls"
url = "https://example.invalid/b.m3u8"

[[camera]]
id = "c"
type = "file"
url = "video.mp4"
""",
    )
    cameras = load_cameras(path)
    assert [c.camera_id for c in cameras] == ["a", "b", "c"]
    assert [c.source_type for c in cameras] == [
        SourceType.MJPEG,
        SourceType.HLS,
        SourceType.FILE,
    ]


def test_example_file_is_valid():
    cameras = load_cameras("cameras.example.toml")
    assert len(cameras) == 3


def test_uses_environment_variable(tmp_path, monkeypatch):
    path = write(tmp_path, '[[camera]]\nid = "a"\ntype = "file"\nurl = "x.mp4"\n')
    monkeypatch.setenv("ATLAS_CAMERAS_FILE", str(path))
    assert load_cameras()[0].camera_id == "a"


def test_missing_path_and_variable(monkeypatch):
    monkeypatch.delenv("ATLAS_CAMERAS_FILE", raising=False)
    with pytest.raises(ValueError, match="ATLAS_CAMERAS_FILE"):
        load_cameras()


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "at least one"),
        ('[[camera]]\ntype = "file"\nurl = "x"\n', "'id' is missing"),
        ('[[camera]]\nid = "a"\ntype = "rtsp"\nurl = "x"\n', "'type' must be one of"),
        ('[[camera]]\nid = "a"\ntype = "file"\n', "'url' is missing"),
        (
            '[[camera]]\nid = "a"\ntype = "file"\nurl = "x"\n'
            '[[camera]]\nid = "a"\ntype = "file"\nurl = "y"\n',
            "used twice",
        ),
    ],
)
def test_invalid_files(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_cameras(write(tmp_path, text))


def test_error_message_does_not_contain_the_address(tmp_path):
    secret = "http://secret-camera.invalid/stream"
    path = write(tmp_path, f'[[camera]]\nid = "a"\ntype = "ftp"\nurl = "{secret}"\n')
    with pytest.raises(ValueError) as error:
        load_cameras(path)
    assert secret not in str(error.value)

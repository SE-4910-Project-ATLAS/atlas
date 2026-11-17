import itertools

import cv2
import numpy as np
import pytest

from app.ingest import CameraConfig, CameraReader, CameraStatus, SourceType


class FakeCapture:
    """Delivers `frame_count` frames, then reports that the stream stopped."""

    def __init__(self, opened=True, frame_count=0, fps=0.0):
        self._opened = opened
        self._remaining = frame_count
        self._fps = fps

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._remaining == 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        pass


def scripted(*captures):
    """An open_capture function that returns the given captures one by one."""
    queue = iter(captures)

    def open_capture(_url):
        return next(queue)

    return open_capture


def stream_config(source_type=SourceType.MJPEG):
    return CameraConfig("cam-1", source_type, "http://example.invalid/stream")


def test_frames_carry_camera_id_and_utc_timestamp():
    reader = CameraReader(
        stream_config(),
        open_capture=scripted(FakeCapture(frame_count=3)),
    )
    frames = list(itertools.islice(reader.frames(), 3))
    assert [f.camera_id for f in frames] == ["cam-1"] * 3
    assert all(f.timestamp.tzinfo is not None for f in frames)
    assert frames[0].timestamp <= frames[1].timestamp <= frames[2].timestamp
    reader.stop()


def test_stream_that_cannot_be_opened_is_offline_then_online():
    events = []
    reader = CameraReader(
        stream_config(),
        on_status=lambda camera_id, status: events.append((camera_id, status)),
        retry_initial=0.0,
        open_capture=scripted(
            FakeCapture(opened=False),
            FakeCapture(opened=False),
            FakeCapture(frame_count=1),
        ),
    )
    frame = next(reader.frames())
    assert frame.camera_id == "cam-1"
    assert events == [
        ("cam-1", CameraStatus.OFFLINE),
        ("cam-1", CameraStatus.ONLINE),
    ]
    assert reader.status is CameraStatus.ONLINE
    reader.stop()


def test_stream_that_stops_is_reported_offline_and_read_again():
    events = []
    reader = CameraReader(
        stream_config(),
        on_status=lambda camera_id, status: events.append(status),
        retry_initial=0.0,
        open_capture=scripted(
            FakeCapture(frame_count=2),
            FakeCapture(frame_count=2),
        ),
    )
    frames = list(itertools.islice(reader.frames(), 4))
    assert len(frames) == 4
    assert events == [
        CameraStatus.ONLINE,
        CameraStatus.OFFLINE,
        CameraStatus.ONLINE,
    ]
    reader.stop()


def test_waiting_time_grows_and_is_capped():
    waits = []
    reader = CameraReader(
        stream_config(),
        retry_initial=1.0,
        retry_max=4.0,
        open_capture=scripted(
            FakeCapture(opened=False),
            FakeCapture(opened=False),
            FakeCapture(opened=False),
            FakeCapture(opened=False),
            FakeCapture(frame_count=1),
        ),
    )
    reader._stop.wait = lambda timeout=None: waits.append(timeout) or False
    next(reader.frames())
    assert waits == [1.0, 2.0, 4.0, 4.0]
    reader.stop()


def test_stop_ends_the_iteration():
    reader = CameraReader(
        stream_config(),
        open_capture=scripted(FakeCapture(frame_count=100)),
    )
    count = 0
    for _ in reader.frames():
        count += 1
        if count == 3:
            reader.stop()
    assert count == 3


def test_file_replay_is_paced_at_the_video_frame_rate():
    ticks = {"now": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        ticks["now"] += seconds

    reader = CameraReader(
        CameraConfig("rec-1", SourceType.FILE, "video.mp4"),
        open_capture=scripted(FakeCapture(frame_count=5, fps=10.0)),
        clock=lambda: ticks["now"],
        sleep=sleep,
    )
    frames = list(reader.frames())
    assert len(frames) == 5  # the iteration ends with the file
    assert sleeps == pytest.approx([0.1] * 4)  # no wait before the first frame


def test_file_replay_does_not_burst_when_running_late():
    ticks = {"now": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        ticks["now"] += seconds

    reader = CameraReader(
        CameraConfig("rec-1", SourceType.FILE, "video.mp4"),
        open_capture=scripted(FakeCapture(frame_count=4, fps=10.0)),
        clock=lambda: ticks["now"],
        sleep=sleep,
    )
    frames = reader.frames()
    next(frames)
    ticks["now"] += 1.0  # the consumer was slow for a whole second
    next(frames)
    next(frames)
    assert sleeps == pytest.approx([0.1])  # only the third frame waits, one interval


def test_missing_video_file_is_offline_not_a_crash(tmp_path):
    events = []
    reader = CameraReader(
        CameraConfig("rec-1", SourceType.FILE, str(tmp_path / "missing.mp4")),
        on_status=lambda camera_id, status: events.append(status),
        retry_initial=0.0,
    )
    reader._stop.wait = lambda timeout=None: reader.stop() or True
    assert list(reader.frames()) == []
    assert events == [CameraStatus.OFFLINE]


def test_reads_a_real_video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for value in range(6):
        writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.release()

    reader = CameraReader(
        CameraConfig("rec-1", SourceType.FILE, str(path)),
        sleep=lambda seconds: None,
    )
    frames = list(reader.frames())
    assert len(frames) == 6
    assert frames[0].image.shape == (48, 64, 3)
    assert reader.status is CameraStatus.ONLINE


def test_default_capture_uses_the_timeouts(monkeypatch):
    calls = []

    def fake_video_capture(url, api, params):
        calls.append((url, api, params))
        return object()

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    reader = CameraReader(
        CameraConfig("cam-1", SourceType.HLS, "http://example.invalid/s"),
        open_timeout_ms=1234,
        read_timeout_ms=5678,
    )
    reader._open_capture("http://example.invalid/s")
    (_, _, params), = calls
    assert params == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1234,
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5678]

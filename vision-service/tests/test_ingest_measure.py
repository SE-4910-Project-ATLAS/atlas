import csv

import numpy as np
import pytest

from app.ingest import CameraConfig, SourceType
from app.ingest.config import load_cameras
from app.ingest.measure import Measurement, append_csv, main, measure, percentile


class ScriptedClock:
    """A fake clock that is moved by the fake capture, not by real time."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TimedCapture:
    """Delivers one frame at each of the given times (seconds), then stops."""

    def __init__(self, clock, times, opened=True):
        self._clock = clock
        self._times = list(times)
        self._opened = opened

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._times:
            return False, None
        self._clock.now = self._times.pop(0)
        return True, np.zeros((36, 64, 3), dtype=np.uint8)

    def get(self, prop):
        return 0.0

    def release(self):
        pass


def stream_config():
    return CameraConfig("cam-1", SourceType.MJPEG, "http://example.invalid/s")


def run(times, *, duration=None, warmup=0.0, source_type=SourceType.MJPEG):
    """Measure a fake stream; it ends with the last frame, which reaches `duration`."""
    clock = ScriptedClock()
    captures = iter([TimedCapture(clock, times)])
    return measure(
        CameraConfig("cam-1", source_type, "http://example.invalid/s"),
        duration=times[-1] if duration is None else duration,
        warmup=warmup,
        clock=clock,
        open_capture=lambda _url: next(captures),
        retry_initial=0.0,
    )


def test_percentile_uses_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 0.95) == 10.0
    assert percentile(values, 0.5) == 5.0
    assert percentile([4.0], 0.95) == 4.0


def test_steady_stream_at_ten_fps():
    times = [round(i * 0.1, 1) for i in range(1, 101)]  # 100 frames, 0.1 s apart
    result = run(times)
    assert result.frames == 100
    assert (result.width, result.height) == (64, 36)
    assert result.first_frame_delay_s == 0.1
    assert result.fps == pytest.approx(10.0, abs=0.01)
    assert result.interval_median_ms == pytest.approx(100.0, abs=1)
    assert result.stalls == 0
    assert result.offline_events == 0


def test_warm_up_frames_are_left_out_of_the_frame_rate():
    # A burst of 50 frames in the first second, as an HLS player reads buffered
    # segments, then a steady 5 fps.
    burst = [i * 0.02 for i in range(1, 51)]
    steady = [1.0 + i * 0.2 for i in range(1, 61)]
    with_warmup = run(burst + steady, warmup=2.0)
    without = run(burst + steady, warmup=0.0)
    assert with_warmup.fps == pytest.approx(5.0, abs=0.05)
    assert without.fps > 8.0  # the burst makes the frame rate look too high


def test_a_long_gap_is_counted_as_a_stall():
    times = [i * 0.1 for i in range(1, 21)] + [5.0] + [5.0 + i * 0.1 for i in range(1, 6)]
    result = run(times)
    assert result.stalls == 1
    assert result.interval_max_ms == pytest.approx(3000.0, abs=1)


def hls_times(segments, *, skip=None):
    """20 frames arrive together every 2 s; segment `skip` (if any) comes 6 s late."""
    times = []
    for n in range(1, segments + 1):
        start = n * 2.0 + (6.0 if skip is not None and n >= skip else 0.0)
        times += [start + i * 0.001 for i in range(20)]
    return times


def test_hls_segment_gaps_are_not_stalls():
    result = run(hls_times(15), source_type=SourceType.HLS)
    assert result.stalls == 0
    assert result.delivery_gap_median_ms == pytest.approx(1981, abs=5)
    assert result.interval_median_ms < 5


def test_hls_segment_that_arrives_late_is_a_stall():
    result = run(hls_times(15, skip=8), source_type=SourceType.HLS)
    assert result.stalls == 1


def test_per_minute_frame_rate_needs_full_minutes():
    short = run([i * 0.5 for i in range(1, 41)])  # 20 s of data
    assert short.fps_min_minute is None
    long = run([i * 0.5 for i in range(1, 241)])  # 120 s at 2 fps
    # The first minute starts before the first frame, so it is one frame short.
    assert long.fps_min_minute == pytest.approx(2.0, abs=0.05)
    assert long.fps_max_minute == pytest.approx(2.0, abs=0.05)


def test_offline_events_are_counted():
    clock = ScriptedClock()
    captures = iter(
        [
            TimedCapture(clock, [1.0, 2.0]),  # stops after two frames
            TimedCapture(clock, [], opened=False),  # cannot be opened
            TimedCapture(clock, [5.0, 6.0]),
        ]
    )
    result = measure(
        stream_config(),
        duration=6.0,
        warmup=0.0,
        clock=clock,
        open_capture=lambda _url: next(captures),
        retry_initial=0.0,
    )
    assert result.frames == 4
    # Two failed attempts in a row are one outage: ONLINE -> OFFLINE -> ONLINE.
    assert result.offline_events == 1


def test_source_without_frames_gives_empty_result():
    clock = ScriptedClock()
    captures = iter([TimedCapture(clock, [])] * 3)

    result = measure(
        stream_config(),
        duration=0.05,  # the timer stops the reader
        warmup=0.0,
        clock=clock,
        open_capture=lambda _url: next(captures, TimedCapture(clock, [], opened=False)),
        retry_initial=0.01,
        retry_max=0.01,
    )
    assert result.frames == 0
    assert result.fps is None
    assert result.first_frame_delay_s is None


def sample(camera_id="cam-1"):
    return Measurement(camera_id, "mjpeg", 600.0, 3000, 640, 360, 0.4, 30.0,
                       5.0, 4.9, 5.1, 200.0, 210.0, 900.0, None, 1, 0)


def test_csv_rows_are_appended_with_one_header(tmp_path):
    path = tmp_path / "out.csv"
    append_csv(path, sample("a"))
    append_csv(path, sample("b"))
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["camera_id"] for r in rows] == ["a", "b"]
    assert rows[0]["fps"] == "5.0"


def test_command_line_reports_unknown_camera(tmp_path, capsys):
    config = tmp_path / "cameras.toml"
    config.write_text('[[camera]]\nid = "a"\ntype = "file"\nurl = "x.mp4"\n', encoding="utf-8")
    assert main(["--config", str(config), "--camera", "zzz"]) == 2
    assert "Unknown camera 'zzz'" in capsys.readouterr().err


def test_command_line_reports_missing_file(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.toml"), "--camera", "a"]) == 2
    assert "Cannot read the camera file" in capsys.readouterr().err


def test_command_line_never_prints_the_address(tmp_path, capsys):
    secret = "http://secret-camera.invalid/stream"
    config = tmp_path / "cameras.toml"
    config.write_text(f'[[camera]]\nid = "a"\ntype = "mjpeg"\nurl = "{secret}"\n', encoding="utf-8")
    main(["--config", str(config), "--camera", "zzz"])
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert load_cameras(config)[0].url == secret

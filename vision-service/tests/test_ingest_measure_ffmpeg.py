import sys
import textwrap

import pytest

from app.ingest.measure_ffmpeg import Sample, measure_with_ffmpeg, summarise
from app.ingest.models import CameraConfig, SourceType

CONFIG = CameraConfig("cam-1", SourceType.HLS, "http://example.invalid/s")


def steady(rate, seconds, *, start=1):
    return [Sample(float(t), (t - start + 1) * rate) for t in range(start, seconds + 1)]


def test_steady_stream_has_the_expected_rates():
    result = summarise(CONFIG, steady(5, 130), duration=130, warmup=10, offline_events=0)
    assert result.fps == pytest.approx(5.0, abs=0.05)
    assert result.fps_min_minute == pytest.approx(5.0, abs=0.05)
    assert result.fps_max_minute == pytest.approx(5.0, abs=0.05)
    assert result.first_frame_delay_s == 1.0
    assert result.stalls == 0
    assert result.frames == 650


def test_a_pause_longer_than_the_limit_is_one_stall():
    samples = steady(5, 20) + [Sample(float(t), 100 + (t - 29) * 5) for t in range(30, 61)]
    result = summarise(CONFIG, samples, duration=60, warmup=0, offline_events=1)
    assert result.stalls == 1
    assert result.offline_events == 1


def test_short_pauses_between_segments_are_not_stalls():
    samples = []
    frames = 0
    for t in range(1, 61):
        if t % 4 == 0:  # a segment of 20 frames every 4 seconds
            frames += 20
        samples.append(Sample(float(t), frames))
    result = summarise(CONFIG, samples, duration=60, warmup=0, offline_events=0)
    assert result.stalls == 0


def test_no_frames_gives_an_empty_result():
    result = summarise(CONFIG, [], duration=60, warmup=10, offline_events=1)
    assert result.frames == 0
    assert result.first_frame_delay_s is None
    assert result.fps == 0.0
    assert result.stalls == 0


def fake_program(tmp_path, body):
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(textwrap.dedent(body))
    return lambda _url: [sys.executable, str(script)]


def test_progress_report_of_the_program_is_counted(tmp_path):
    build = fake_program(
        tmp_path,
        """
        import sys, time
        print("  Stream #0:0: Video: h264, yuv420p, 640x360, 10 fps", file=sys.stderr, flush=True)
        for n in range(1, 40):
            print(f"frame={n * 10}", flush=True)
            print("progress=continue", flush=True)
            time.sleep(0.1)
        """,
    )
    timeline = []
    result = measure_with_ffmpeg(
        CONFIG, duration=2.0, warmup=0.5, build_command=build, retry_initial=0.1, timeline=timeline
    )
    assert result.frames > 100
    assert (result.width, result.height) == (640, 360)
    assert result.fps > 50
    assert result.offline_events == 0
    assert timeline and timeline[0][1] == "online"


def test_a_program_that_stops_is_started_again_and_counted_once(tmp_path):
    build = fake_program(tmp_path, "import sys\nsys.exit(1)\n")
    timeline = []
    result = measure_with_ffmpeg(
        CONFIG, duration=1.5, warmup=0.0, build_command=build, retry_initial=0.1, retry_max=0.2, timeline=timeline
    )
    assert result.frames == 0
    assert result.offline_events == 1
    assert [state for _, state in timeline] == ["offline"]

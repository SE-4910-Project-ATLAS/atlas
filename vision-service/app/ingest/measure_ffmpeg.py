"""Measure a stream with the FFmpeg program instead of OpenCV (S4PT-50).

    python -m app.ingest.measure_ffmpeg --config cameras.local.toml --camera ibb-004 \
        --duration 600 --csv measurements.csv

Why this tool exists: on 17.11.2026 the FFmpeg that is built into OpenCV was
refused in the TLS handshake by the İBB HLS server ("Failed to read handshake
response"), while a separate FFmpeg program could read the same stream. This tool
reads the stream with that program (it must be on the PATH) and counts the frames
from its progress report, which FFmpeg prints once per second.

It is a measurement tool only. The stream reader of the vision service
(`app.ingest.reader`) does not use it.

Differences from `app.ingest.measure`: the progress report gives the number of
frames per second, not the arrival time of every frame, so the frame intervals
are not reported. The frame rate, the frame rate per full minute, the stalls (no
new frame for STALL_S seconds), the first frame delay and the offline events are
reported in the same way. The end-to-end delay cannot be measured (see
`app.ingest.measure`).
"""

import argparse
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.ingest.config import load_cameras
from app.ingest.measure import (
    DEFAULT_DURATION_S,
    DEFAULT_WARMUP_S,
    Measurement,
    append_csv,
    format_result,
)
from app.ingest.models import CameraConfig

STALL_S = 6.0  # no new frame for this long counts as one stall
RETRY_INITIAL_S = 1.0
RETRY_MAX_S = 30.0
_SIZE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")


@dataclass(frozen=True)
class Sample:
    t: float  # seconds since the start of the measurement
    frames: int  # frames received so far, over all restarts of FFmpeg


def ffmpeg_command(url: str, ffmpeg: str = "ffmpeg") -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "info", "-nostats",
        "-progress", "pipe:1", "-stats_period", "1",
        "-i", url, "-an", "-fps_mode", "passthrough", "-f", "null", "-",
    ]  # fmt: skip


def _frames_at(samples: Sequence[Sample], moment: float) -> int:
    """Frames received up to `moment` (the last report before it)."""
    count = 0
    for sample in samples:
        if sample.t > moment:
            break
        count = sample.frames
    return count


def summarise(
    config: CameraConfig,
    samples: Sequence[Sample],
    *,
    duration: float,
    warmup: float,
    offline_events: int,
    width: int = 0,
    height: int = 0,
) -> Measurement:
    first = next((s.t for s in samples if s.frames > 0), None)

    fps = None
    if duration > warmup:
        fps = (_frames_at(samples, duration) - _frames_at(samples, warmup)) / (duration - warmup)

    minutes: list[float] = []
    start = warmup
    while start + 60 <= duration + 1e-9:
        minutes.append((_frames_at(samples, start + 60) - _frames_at(samples, start)) / 60)
        start += 60

    stalls = 0
    last_new = first
    previous = 0
    for sample in samples:
        if sample.frames > previous:
            if last_new is not None and sample.t - last_new > STALL_S:
                stalls += 1
            last_new = sample.t
            previous = sample.frames
    if last_new is not None and duration - last_new > STALL_S:
        stalls += 1

    def rounded(value: float | None, digits: int = 2) -> float | None:
        return None if value is None else round(value, digits)

    return Measurement(
        camera_id=config.camera_id,
        source_type=config.source_type.value,
        duration_s=round(duration, 1),
        frames=samples[-1].frames if samples else 0,
        width=width,
        height=height,
        first_frame_delay_s=rounded(first),
        warmup_s=warmup,
        fps=rounded(fps),
        fps_min_minute=rounded(min(minutes)) if minutes else None,
        fps_max_minute=rounded(max(minutes)) if minutes else None,
        interval_median_ms=None,
        interval_p95_ms=None,
        interval_max_ms=None,
        delivery_gap_median_ms=None,
        stalls=stalls,
        offline_events=offline_events,
    )


def measure_with_ffmpeg(
    config: CameraConfig,
    *,
    duration: float = DEFAULT_DURATION_S,
    warmup: float = DEFAULT_WARMUP_S,
    build_command: Callable[[str], list[str]] | None = None,
    retry_initial: float = RETRY_INITIAL_S,
    retry_max: float = RETRY_MAX_S,
    clock: Callable[[], float] = time.monotonic,
    timeline: list[tuple[float, str]] | None = None,
) -> Measurement:
    build = build_command or ffmpeg_command
    started = clock()
    samples: list[Sample] = []
    offline_events = 0
    state: str | None = None
    base_frames = 0
    size = (0, 0)
    delay = retry_initial

    def note(new_state: str) -> None:
        nonlocal state, offline_events
        if new_state == state:
            return
        state = new_state
        if new_state == "offline":
            offline_events += 1
        if timeline is not None:
            timeline.append((round(clock() - started, 1), new_state))

    while clock() - started < duration:
        process = subprocess.Popen(  # noqa: S603 - the command is built from our own configuration
            build(config.url), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_progress(stream=process.stdout, out=lines) -> None:
            for line in stream:
                out.put(line.strip())
            out.put(None)

        def read_log(stream=process.stderr) -> None:
            nonlocal size
            for line in stream:
                if "Video:" in line and size == (0, 0):
                    match = _SIZE.search(line.split("Video:", 1)[1])
                    if match:
                        size = (int(match.group(1)), int(match.group(2)))

        threading.Thread(target=read_progress, daemon=True).start()
        threading.Thread(target=read_log, daemon=True).start()

        frame = 0
        ended = False
        while clock() - started < duration:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                ended = True
                break
            if line.startswith("frame="):
                frame = int(line.split("=", 1)[1] or 0)
            elif line.startswith("progress="):
                samples.append(Sample(round(clock() - started, 3), base_frames + frame))
                if frame > 0:
                    note("online")
                    delay = retry_initial

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        base_frames += frame

        if ended and clock() - started < duration:
            note("offline")
            time.sleep(min(delay, max(0.0, duration - (clock() - started))))
            delay = min(delay * 2, retry_max)

    return summarise(
        config, samples, duration=clock() - started, warmup=warmup,
        offline_events=offline_events, width=size[0], height=size[1],
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a camera stream with the FFmpeg program.")
    parser.add_argument("--config", help="camera TOML file (default: $ATLAS_CAMERAS_FILE)")
    parser.add_argument("--camera", required=True, help="id of the camera to measure")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="seconds to measure")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S, help="seconds left out of the statistics")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg program (default: ffmpeg on the PATH)")
    parser.add_argument("--csv", help="append the result as one row to this CSV file")
    args = parser.parse_args(argv)

    try:
        cameras = load_cameras(args.config)
    except (ValueError, OSError) as error:
        print(f"Cannot read the camera file: {error}", file=sys.stderr)
        return 2
    by_id = {camera.camera_id: camera for camera in cameras}
    if args.camera not in by_id:
        print(f"Unknown camera '{args.camera}'. Known cameras: {', '.join(sorted(by_id))}", file=sys.stderr)
        return 2

    print(f"Measuring camera {args.camera} with FFmpeg for {args.duration:g} s ...", file=sys.stderr)
    timeline: list[tuple[float, str]] = []
    try:
        result = measure_with_ffmpeg(
            by_id[args.camera],
            duration=args.duration,
            warmup=args.warmup,
            build_command=lambda url: ffmpeg_command(url, args.ffmpeg),
            timeline=timeline,
        )
    except FileNotFoundError:
        print(f"FFmpeg program '{args.ffmpeg}' not found. Install FFmpeg or pass --ffmpeg.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130

    print(format_result(result))
    if timeline:
        print("status changes: " + ", ".join(f"{state} at {seconds:g} s" for seconds, state in timeline[:20]))
    if args.csv:
        append_csv(args.csv, result)
        print(f"Result added to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

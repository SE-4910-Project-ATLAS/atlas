"""Measure the real frame rate of one camera source.

    python -m app.ingest.measure --config cameras.local.toml --camera izum-001 \
        --duration 600 --csv measurements.csv

What is measured (all times come from the local monotonic clock):

- first frame delay: time from opening the source to the first frame. For an
  HLS stream this includes the buffering of the first segments.
- frame rate and frame intervals after a warm-up period. An HLS player first
  reads the segments that are already buffered, faster than real time, so the
  first seconds would make the frame rate look too high. They are left out.
- stalls: gaps between two frames that are much longer than usual. An HLS
  player hands over a whole segment of frames at once and then waits for the
  next segment, so its frame intervals are ~0 ms inside a segment and about one
  segment length between segments. For HLS the usual gap is therefore the
  segment gap (reported as delivery gap) and a stall is a gap clearly longer
  than that.
- offline events: how often the stream stopped and was opened again.

The end-to-end delay (from the camera sensor to our frame) cannot be measured
this way, because a third-party stream carries no reference clock. Only the
first frame delay is reported.
"""

import argparse
import csv
import statistics
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from app.ingest.config import load_cameras
from app.ingest.models import CameraConfig, CameraStatus, SourceType
from app.ingest.reader import CameraReader

DEFAULT_DURATION_S = 600.0  # the task asks for at least 10 minutes
DEFAULT_WARMUP_S = 30.0
STALL_FACTOR = 5.0  # a gap longer than this many median intervals is a stall
STALL_MIN_S = 1.0  # ... and at least this long
DELIVERY_GAP_S = 0.25  # HLS: a gap longer than this is a wait for the next segment
HLS_STALL_FACTOR = 2.5  # HLS: a stall is longer than this many usual delivery gaps


@dataclass(frozen=True)
class Measurement:
    camera_id: str
    source_type: str
    duration_s: float  # how long the measurement ran
    frames: int  # frames received in total
    width: int
    height: int
    first_frame_delay_s: float | None
    warmup_s: float
    fps: float | None  # frames per second after the warm-up
    fps_min_minute: float | None  # slowest full minute after the warm-up
    fps_max_minute: float | None  # fastest full minute after the warm-up
    interval_median_ms: float | None
    interval_p95_ms: float | None
    interval_max_ms: float | None
    delivery_gap_median_ms: float | None  # HLS only: usual wait between segments
    stalls: int
    offline_events: int


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    index = max(0, min(len(sorted_values) - 1, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def measure(
    config: CameraConfig,
    *,
    duration: float = DEFAULT_DURATION_S,
    warmup: float = DEFAULT_WARMUP_S,
    clock: Callable[[], float] = time.monotonic,
    timeline: list[tuple[float, str]] | None = None,
    **reader_options: Any,
) -> Measurement:
    """Read the source for `duration` seconds and summarise what arrived.

    `reader_options` are passed to CameraReader (for example `open_capture`).
    When a list is given as `timeline`, every status change is added to it as
    (seconds since the start, "online" or "offline").
    """
    offline_events = 0
    started = clock()

    def on_status(_camera_id: str, status: CameraStatus) -> None:
        nonlocal offline_events
        if status is CameraStatus.OFFLINE:
            offline_events += 1
        if timeline is not None:
            timeline.append((round(clock() - started, 1), status.value))

    reader = CameraReader(config, on_status=on_status, clock=clock, **reader_options)

    # A source that never delivers a frame must not run for ever: the timer stops
    # the reader after `duration` real seconds, also while it waits to reconnect.
    timer = threading.Timer(duration, reader.stop)
    timer.daemon = True
    timer.start()

    arrivals: list[float] = []  # seconds since the start
    shape = (0, 0)
    try:
        for frame in reader.frames():
            elapsed = clock() - started
            if not arrivals:
                shape = frame.image.shape[:2]
            arrivals.append(elapsed)
            if elapsed >= duration:
                reader.stop()
                break
    finally:
        timer.cancel()
    total = clock() - started

    steady = [t for t in arrivals if t >= warmup]
    fps = None
    intervals: list[float] = []
    if len(steady) >= 2:
        intervals = [b - a for a, b in zip(steady, steady[1:])]
        span = steady[-1] - steady[0]
        fps = (len(steady) - 1) / span if span > 0 else None

    fps_min = fps_max = None
    full_minutes = int((arrivals[-1] - warmup) // 60) if arrivals else 0
    if full_minutes >= 1:
        per_minute = []
        for minute in range(full_minutes):
            low = warmup + minute * 60
            per_minute.append(sum(1 for t in arrivals if low <= t < low + 60) / 60.0)
        fps_min, fps_max = min(per_minute), max(per_minute)

    median = p95 = longest = delivery_gap = None
    stalls = 0
    if intervals:
        ordered = sorted(intervals)
        median = statistics.median(ordered)
        p95 = percentile(ordered, 0.95)
        longest = ordered[-1]
        limit = max(STALL_MIN_S, STALL_FACTOR * median)
        if config.source_type is SourceType.HLS:
            gaps = [value for value in intervals if value > DELIVERY_GAP_S]
            if gaps:
                delivery_gap = statistics.median(gaps)
                limit = max(STALL_MIN_S, HLS_STALL_FACTOR * delivery_gap)
        stalls = sum(1 for value in intervals if value > limit)

    def ms(value: float | None) -> float | None:
        return None if value is None else round(value * 1000, 1)

    def rounded(value: float | None, digits: int = 2) -> float | None:
        return None if value is None else round(value, digits)

    return Measurement(
        camera_id=config.camera_id,
        source_type=config.source_type.value,
        duration_s=round(total, 1),
        frames=len(arrivals),
        width=shape[1],
        height=shape[0],
        first_frame_delay_s=rounded(arrivals[0]) if arrivals else None,
        warmup_s=warmup,
        fps=rounded(fps),
        fps_min_minute=rounded(fps_min),
        fps_max_minute=rounded(fps_max),
        interval_median_ms=ms(median),
        interval_p95_ms=ms(p95),
        interval_max_ms=ms(longest),
        delivery_gap_median_ms=ms(delivery_gap),
        stalls=stalls,
        offline_events=offline_events,
    )


def append_csv(path: str | Path, result: Measurement) -> None:
    """Add one row to a CSV file; the header is written when the file is new."""
    path = Path(path)
    new_file = not path.exists() or path.stat().st_size == 0
    names = [f.name for f in fields(Measurement)]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(result))


def format_result(result: Measurement) -> str:
    rows: list[tuple[str, Any]] = [
        ("camera", result.camera_id),
        ("source type", result.source_type),
        ("measured for (s)", result.duration_s),
        ("frames received", result.frames),
        ("frame size", f"{result.width}x{result.height}"),
        ("first frame delay (s)", result.first_frame_delay_s),
        (f"frame rate after {result.warmup_s:g} s warm-up (fps)", result.fps),
        ("slowest full minute (fps)", result.fps_min_minute),
        ("fastest full minute (fps)", result.fps_max_minute),
        ("frame interval, median (ms)", result.interval_median_ms),
        ("frame interval, 95th percentile (ms)", result.interval_p95_ms),
        ("frame interval, longest (ms)", result.interval_max_ms),
        ("HLS delivery gap, median (ms)", result.delivery_gap_median_ms),
        ("stalls", result.stalls),
        ("offline events", result.offline_events),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name:<{width}}  {value if value is not None else '-'}" for name, value in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the real frame rate of a camera source.")
    parser.add_argument("--config", help="camera TOML file (default: $ATLAS_CAMERAS_FILE)")
    parser.add_argument("--camera", required=True, help="id of the camera to measure")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="seconds to measure")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S, help="seconds left out of the statistics")
    parser.add_argument("--csv", help="append the result as one row to this CSV file")
    args = parser.parse_args(argv)

    try:
        cameras = load_cameras(args.config)
    except (ValueError, OSError) as error:
        print(f"Cannot read the camera file: {error}", file=sys.stderr)
        return 2

    by_id = {camera.camera_id: camera for camera in cameras}
    if args.camera not in by_id:
        known = ", ".join(sorted(by_id))
        print(f"Unknown camera '{args.camera}'. Known cameras: {known}", file=sys.stderr)
        return 2

    print(f"Measuring camera {args.camera} for {args.duration:g} s ...", file=sys.stderr)
    timeline: list[tuple[float, str]] = []
    try:
        result = measure(by_id[args.camera], duration=args.duration, warmup=args.warmup, timeline=timeline)
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

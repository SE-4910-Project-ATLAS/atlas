"""Read frames from one camera source with offline detection and reconnection."""

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import cv2

from app.ingest.models import CameraConfig, CameraStatus, Frame, SourceType

logger = logging.getLogger(__name__)

DEFAULT_FILE_FPS = 25.0  # used when a video file does not report its frame rate

StatusCallback = Callable[[str, CameraStatus], None]


class CameraReader:
    """Yields the frames of one camera.

    - Every frame gets the camera ID and a UTC timestamp.
    - A video file is replayed at real-time speed, as a simulated live stream,
      and the iteration ends when the file ends.
    - A stream that cannot be opened, or stops delivering frames, is reported as
      OFFLINE and opened again with a growing wait time. When frames arrive
      again it is reported as ONLINE.
    """

    def __init__(
        self,
        config: CameraConfig,
        *,
        on_status: StatusCallback | None = None,
        retry_initial: float = 1.0,
        retry_max: float = 30.0,
        open_capture: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._on_status = on_status
        self._retry_initial = retry_initial
        self._retry_max = retry_max
        self._open_capture = open_capture or cv2.VideoCapture
        self._clock = clock
        self._sleep = sleep
        self._stop = threading.Event()
        self._status: CameraStatus | None = None

    @property
    def status(self) -> CameraStatus | None:
        return self._status

    def stop(self) -> None:
        """Ask the reader to finish; it stops at the next frame or wait."""
        self._stop.set()

    def frames(self) -> Iterator[Frame]:
        delay = self._retry_initial
        while not self._stop.is_set():
            capture = self._open_capture(self._config.url)
            if not capture.isOpened():
                capture.release()
                self._set_status(CameraStatus.OFFLINE)
                self._stop.wait(delay)
                delay = min(delay * 2, self._retry_max)
                continue

            is_file = self._config.source_type is SourceType.FILE
            interval = self._frame_interval(capture) if is_file else None
            next_due = self._clock()
            received_any = False

            while not self._stop.is_set():
                ok, image = capture.read()
                if not ok or image is None:
                    break
                if not received_any:
                    received_any = True
                    delay = self._retry_initial
                    self._set_status(CameraStatus.ONLINE)
                if interval is not None:
                    # Pace the replay; do not burst to catch up when running late.
                    now = self._clock()
                    if next_due > now:
                        self._sleep(next_due - now)
                    next_due = max(next_due, now) + interval
                yield Frame(self._config.camera_id, datetime.now(UTC), image)

            capture.release()
            if is_file:
                return  # the end of a recorded video is not a failure
            self._set_status(CameraStatus.OFFLINE)
            self._stop.wait(delay)
            delay = min(delay * 2, self._retry_max)

    def _frame_interval(self, capture: Any) -> float:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = DEFAULT_FILE_FPS
        return 1.0 / fps

    def _set_status(self, status: CameraStatus) -> None:
        if status is self._status:
            return
        self._status = status
        logger.info("Camera %s is %s", self._config.camera_id, status.value)
        if self._on_status is not None:
            self._on_status(self._config.camera_id, status)

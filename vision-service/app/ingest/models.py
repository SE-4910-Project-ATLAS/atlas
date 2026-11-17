from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np


class SourceType(str, Enum):
    """Kinds of camera source the reader can open."""

    MJPEG = "mjpeg"  # Izum MJPEG streams
    HLS = "hls"  # IBB HLS streams
    FILE = "file"  # recorded video replayed at real-time speed


class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source_type: SourceType
    # Stream address, or the path of the video file. Never log this value:
    # stream addresses must not appear in logs or in Git.
    url: str


@dataclass(frozen=True)
class Frame:
    camera_id: str
    timestamp: datetime  # UTC time at which the frame was read
    image: np.ndarray  # BGR image, as returned by OpenCV

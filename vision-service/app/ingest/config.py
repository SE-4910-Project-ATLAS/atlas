"""Camera list configuration.

The cameras are read from a TOML file, not from the code. The real file holds
stream addresses and must stay outside Git; the repository only contains
`cameras.example.toml` with placeholders.
"""

import os
import tomllib
from pathlib import Path

from app.ingest.models import CameraConfig, SourceType

CAMERAS_FILE_ENV = "ATLAS_CAMERAS_FILE"


def load_cameras(path: str | os.PathLike[str] | None = None) -> list[CameraConfig]:
    """Read the camera list. Without `path`, use the file named in ATLAS_CAMERAS_FILE."""
    if path is None:
        path = os.environ.get(CAMERAS_FILE_ENV)
        if not path:
            raise ValueError(f"No camera file given and {CAMERAS_FILE_ENV} is not set")

    with Path(path).open("rb") as f:
        data = tomllib.load(f)

    entries = data.get("camera", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("The camera file must contain at least one [[camera]] entry")

    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        camera_id = entry.get("id")
        source_type = entry.get("type")
        url = entry.get("url")

        if not camera_id or not isinstance(camera_id, str):
            raise ValueError(f"Camera entry {index}: 'id' is missing")
        if camera_id in seen:
            raise ValueError(f"Camera id '{camera_id}' is used twice")
        seen.add(camera_id)

        try:
            parsed_type = SourceType(source_type)
        except ValueError:
            allowed = ", ".join(t.value for t in SourceType)
            raise ValueError(
                f"Camera '{camera_id}': 'type' must be one of: {allowed}"
            ) from None

        # The address is deliberately not repeated in error messages.
        if not url or not isinstance(url, str):
            raise ValueError(f"Camera '{camera_id}': 'url' is missing")

        cameras.append(CameraConfig(camera_id, parsed_type, url))

    return cameras

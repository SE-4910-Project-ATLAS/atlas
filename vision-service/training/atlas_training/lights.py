"""Flashing red / blue light evidence for a tracked vehicle.

An ambulance has right of way only with its light and sound signals on together
(Geçiş Üstünlüğüne Sahip Araçlar Hakkında Yönetmelik, based on 2918 KTK art. 71), and
Turkish ambulance beacons flash red and blue fast, one after the other or together. The
false alarms we see are vehicles under steady light: parked cars under a blue shop sign,
a truck with red advertising, white vans. So the *change* of lit red and blue pixels over
about a second separates them, not the amount (Fan et al., arXiv 2212.13696: per-frame
light evidence smoothed over time; frequency matching patents, e.g. US11341753).

Per frame: the fraction of the vehicle box that is lit red and lit blue (bright and
saturated). Per track: the largest range (max - min) of either fraction within any
`window` seconds. At 5 frames/s a 2-4 Hz beacon is sampled irregularly (aliasing), but the
range still shows on and off states, which a frequency estimate would not.
"""

from __future__ import annotations

import numpy as np

# OpenCV HSV: hue 0-180. Thresholds set on labelled ambulance boxes (22.11.2026): in
# compressed CCTV a beacon core is nearly white, only its halo is coloured, so stricter
# values (S >= 120, V >= 200) found lit pixels in 0-46 % of the boxes.
MIN_VALUE = 160
MIN_SATURATION = 80
RED_HUES = ((0, 10), (170, 180))
BLUE_HUES = ((95, 135),)


def lit_fractions(image: np.ndarray, box) -> tuple[float, float]:
    """Fractions of the box (x1, y1, x2, y2, BGR image) that are lit red and lit blue."""
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)

    def share(ranges) -> float:
        mask = np.zeros_like(bright)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue <= hi)
        return float(np.mean(bright & mask))

    return share(RED_HUES), share(BLUE_HUES)


def flicker(times: list[float], red: list[float], blue: list[float], window: float = 1.0) -> list[float]:
    """For each frame of one track: the largest range of the red or blue fraction over the
    frames of the last `window` seconds (0 for the first frame)."""
    out = []
    start = 0
    for i, t in enumerate(times):
        while times[start] < t - window:
            start += 1
        r, b = red[start:i + 1], blue[start:i + 1]
        out.append(max(max(r) - min(r), max(b) - min(b)))
    return out


def alarm_with_lights(times: list[float], frames: list[dict], lights: list[dict], min_frames: int,
                      threshold: float, strict: float, min_flicker: float, window: float = 1.0) -> list[bool]:
    """The mean rule with light evidence: a track whose lit red or blue share has changed by
    at least `min_flicker` within `window` seconds (so far) alarms at mean >= `threshold`;
    a track that has not flashed needs mean >= `strict`. `strict` > 1 makes the light a
    requirement (only ambulances on duty have right of way)."""
    total: dict[int, float] = {}
    count: dict[int, int] = {}
    history: dict[int, tuple[list[float], list[float], list[float]]] = {}
    flashed: set[int] = set()
    alarm = []
    for t, probs, lit in zip(times, frames, lights, strict=True):
        on = False
        for track_id, p in probs.items():
            total[track_id] = total.get(track_id, 0.0) + p
            count[track_id] = count.get(track_id, 0) + 1
            fr = lit.get(track_id) or lit.get(str(track_id))
            if fr is not None and track_id not in flashed:
                ts, rs, bs = history.setdefault(track_id, ([], [], []))
                ts.append(t); rs.append(fr[0]); bs.append(fr[1])
                while ts[0] < t - window:
                    ts.pop(0); rs.pop(0); bs.pop(0)
                if max(max(rs) - min(rs), max(bs) - min(bs)) >= min_flicker:
                    flashed.add(track_id)
                    del history[track_id]
            n = count[track_id]
            if n >= min_frames and total[track_id] / n >= (threshold if track_id in flashed else strict):
                on = True
        alarm.append(on)
    return alarm

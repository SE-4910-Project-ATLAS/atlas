import numpy as np

from atlas_training.lights import alarm_with_lights, flicker, lit_fractions


def test_lit_fractions_counts_bright_red_and_blue():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5, :, :] = (0, 0, 255)      # bright red (BGR), top half
    image[5:, :5, :] = (255, 0, 0)     # bright blue, a quarter
    red, blue = lit_fractions(image, (0, 0, 10, 10))
    assert abs(red - 0.5) < 1e-6 and abs(blue - 0.25) < 1e-6


def test_white_and_dark_pixels_are_not_lit():
    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert lit_fractions(image, (0, 0, 10, 10)) == (0.0, 0.0)


def test_flicker_is_the_range_within_the_window():
    times = [0.0, 0.2, 0.4, 2.0]
    assert flicker(times, [0.0, 0.3, 0.1, 0.1], [0.0] * 4) == [0.0, 0.3, 0.3, 0.0]


def test_steady_light_needs_the_strict_threshold_and_flashing_does_not():
    times = [i * 0.2 for i in range(10)]
    frames = [{1: 0.7, 2: 0.7}] * 10
    lights = [{1: [0.0, 0.2], 2: [0.0, 0.2 if i % 2 else 0.0]} for i in range(10)]
    on = alarm_with_lights(times, frames, lights, 5, 0.5, strict=0.9, min_flicker=0.05)
    assert on[-1]
    steady_only = [{1: f[1]} for f in frames]
    assert not any(alarm_with_lights(times, steady_only, lights, 5, 0.5, strict=0.9, min_flicker=0.05))


def test_place_rule_with_lights_is_stricter_for_steady_tracks():
    from atlas_training.tracks import alarm_once_per_place

    times = [i * 0.2 for i in range(10)]
    frames = [{1: 0.7}] * 10
    boxes = [{1: [100, 100, 50]}] * 10
    steady = [{1: [0.0, 0.2]}] * 10
    flashing = [{1: [0.0, 0.2 if i % 2 else 0.0]} for i in range(10)]
    assert not any(alarm_once_per_place(times, frames, boxes, 5, 0.5, lights=steady, strict=0.8, min_flicker=0.05))
    assert alarm_once_per_place(times, frames, boxes, 5, 0.5, lights=flashing, strict=0.8, min_flicker=0.05)[-1]

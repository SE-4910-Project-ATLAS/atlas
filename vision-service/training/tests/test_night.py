import numpy as np

from atlas_training.night import nightify


def test_nightify_darkens_and_keeps_shape():
    rng = np.random.default_rng(0)
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    out = nightify(image, rng)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out.mean() < image.mean() / 2


def test_bright_lights_stay_bright():
    rng = np.random.default_rng(1)
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    image[:4, :4] = 255
    out = nightify(image, rng)
    assert out[:4, :4].mean() > 180 and out[8:, 8:].mean() < 60

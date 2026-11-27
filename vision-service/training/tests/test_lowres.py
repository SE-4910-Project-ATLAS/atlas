import numpy as np

from atlas_training.lowres import degrade


def test_degrade_keeps_size_and_removes_detail():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    out = degrade(image, 32)
    assert out.shape == image.shape
    assert out.astype(float).std() < image.astype(float).std()

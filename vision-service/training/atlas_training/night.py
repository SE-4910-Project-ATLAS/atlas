"""Cheap day-to-night copies of classifier crops (a supplement, not a replacement).

Only about 3 % of the ambulance crops of classifier v5e are dark, and the night clip of the
development set is where the classifier is weakest. Learned day-to-night translation is
the method with published evidence (GAN style transfer: Lin et al., IEEE T-ITS 2021,
arXiv 2105.05011; Yang et al., arXiv 2412.16478); this module is the cheap stand-in:
gamma darkening, a cool colour cast, sensor noise and slight blur. Whether it helps is
measured, not assumed: a classifier is trained with and without these copies and the two
are compared on the night frames of the development set. Real night crops (tr3, mined
along tracks) come first.
"""

from __future__ import annotations

import numpy as np


def nightify(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A darker, noisier, bluish copy of a BGR crop."""
    import cv2

    img = image.astype(np.float32) / 255.0
    gamma = rng.uniform(2.0, 3.2)
    img = img ** gamma                                   # dark, but highlights (lights) stay bright
    cast = np.array([rng.uniform(1.0, 1.25), 1.0, rng.uniform(0.75, 0.95)], dtype=np.float32)  # B, G, R
    img = np.clip(img * cast, 0, 1)
    sigma = rng.uniform(0.01, 0.04)
    img = np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)
    out = (img * 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out

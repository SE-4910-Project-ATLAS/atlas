import numpy as np

from atlas_training.calibrate import apply_temperature, ece, fit_temperature


def test_temperature_keeps_order_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(float)
    over = apply_temperature(true_p, 1 / 3)  # sharpened, over-confident
    t = fit_temperature(over, y)
    fixed = apply_temperature(over, t)
    assert 2.0 < t < 4.5
    assert ece(fixed, y) < ece(over, y)
    assert np.all(np.diff(fixed[np.argsort(over)]) >= -1e-12)

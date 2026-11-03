"""The synthetic test bench must keep the properties the experiments rely on."""
from __future__ import annotations

import numpy as np
import pytest

from bench import NOISE_EFF, NOISE_TEMP, T_LIMIT, efficiency, make_dataset, temperature, to_units


def test_grid_shape_and_range(ff):
    assert ff.X.shape == (41 * 41, 2)
    assert ff.X.min() == 0.0 and ff.X.max() == 1.0
    assert len(np.unique(ff.X, axis=0)) == len(ff.X)


def test_units_cover_engine_operating_range():
    speed, load = to_units(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert speed.tolist() == [1000.0, 6000.0]
    assert load.tolist() == [0.0, 100.0]


def test_safe_region_is_a_realistic_share(ff):
    # About three quarters safe: enough room to explore, enough limit to matter.
    assert 0.65 < ff.safe_true.mean() < 0.85


def test_low_speed_low_load_corner_is_safe(ff):
    assert temperature(np.array([0.0, 0.0])) < T_LIMIT - 100


def test_unconstrained_peak_lies_beyond_the_limit(ff):
    """The tempting efficiency peak must be unsafe, otherwise safety would not matter."""
    peak = np.argmax(ff.f_true)
    assert ff.T_true[peak] > T_LIMIT


def test_best_safe_point_sits_on_the_boundary(ff):
    best = np.argmax(np.where(ff.safe_true, ff.f_true, -np.inf))
    assert T_LIMIT - 30 < ff.T_true[best] <= T_LIMIT


def test_measurement_noise_has_the_stated_level():
    big = make_dataset(n_side=81, seed=7)
    assert np.std(big.f_meas - big.f_true) == pytest.approx(NOISE_EFF, rel=0.1)
    assert np.std(big.T_meas - big.T_true) == pytest.approx(NOISE_TEMP, rel=0.1)


def test_dataset_is_reproducible_per_seed():
    a, b, c = make_dataset(seed=3), make_dataset(seed=3), make_dataset(seed=4)
    np.testing.assert_array_equal(a.f_meas, b.f_meas)
    assert not np.allclose(a.f_meas, c.f_meas)


def test_true_functions_are_finite_and_vectorised():
    X = np.random.default_rng(0).random((5, 3, 2))       # any leading shape
    assert efficiency(X).shape == (5, 3)
    assert np.isfinite(temperature(X)).all()

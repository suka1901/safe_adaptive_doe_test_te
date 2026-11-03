"""Sampling strategies: correctness of the baselines and the safety guarantees of the GP methods."""
from __future__ import annotations

import numpy as np
import pytest

import strategies as st
from bench import T_LIMIT


# ---------- baselines -------------------------------------------------------

def test_grid_order_is_unique_and_respects_budget(ff, rng):
    h = st.grid_order(ff, 50, rng)
    assert len(h.order) == 50
    assert len(set(h.order)) == 50


def test_grid_starts_with_the_domain_corners(ff, rng):
    first4 = {tuple(ff.X[i]) for i in st.grid_order(ff, 4, rng).order}
    assert first4 == {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}


def test_random_order_is_unique_and_reproducible(ff):
    a = st.random_order(ff, 30, np.random.default_rng(5)).order
    b = st.random_order(ff, 30, np.random.default_rng(5)).order
    assert a == b and len(set(a)) == 30


def test_seed_points_are_truly_safe(ff):
    seeds = st.seed_points(ff)
    assert len(seeds) == 4
    assert (ff.T_true[seeds] < T_LIMIT).all()


# ---------- safe-set certification -----------------------------------------

def test_safe_set_applies_confidence_bound_and_margin():
    mu = np.array([800.0, 870.0, 880.0])
    sd = np.array([10.0, 10.0, 0.0])
    ok = st.safe_set(mu, sd, beta=2.0)
    # 800+20=820 ok; 870+20=890 is exactly limit-margin -> ok; 880 ok
    assert ok.tolist() == [True, True, True]
    ok = st.safe_set(mu + 1.0, sd, beta=2.0)
    assert ok.tolist() == [True, False, True]


def test_safe_set_never_jumps_far_from_measured_safe_points():
    """The SafeOpt-style reach rule: a confident-looking but distant point is not certified."""
    X = np.array([[0.0, 0.0], [0.05, 0.0], [0.9, 0.9]])
    mu, sd = np.full(3, 600.0), np.full(3, 1.0)
    ok = st.safe_set(mu, sd, beta=2.5, X=X, safe_measured=np.array([0]))
    assert ok.tolist() == [True, True, False]


def test_measured_safe_uses_measurements(ff):
    hot = int(np.argmax(ff.T_meas))
    cold = int(np.argmin(ff.T_meas))
    assert st._measured_safe(ff, [hot, cold]).tolist() == [cold]


# ---------- GP strategies: the safety guarantees ---------------------------

@pytest.fixture(scope="module")
def active_run(ff_small):
    return st.safe_active(ff_small, 30, np.random.default_rng(0), beta=2.5, seed=0)


@pytest.fixture(scope="module")
def safe_bo_run(ff):
    # full 41 x 41 resolution: the best safe point sits right on the boundary
    return st.safe_bo(ff, 40, np.random.default_rng(0), beta=2.5, seed=0)


def test_safe_active_learning_never_trips(ff_small, active_run):
    assert (ff_small.T_true[active_run.order] <= T_LIMIT).all()


def test_safe_active_learning_explores_beyond_the_seeds(ff_small, active_run):
    pts = ff_small.X[active_run.order]
    assert len(active_run.order) == 30
    assert len(set(active_run.order)) == 30
    assert pts[:, 0].max() > 0.5 and pts[:, 1].max() > 0.5


def test_safe_bo_never_trips_and_gets_close_to_the_best_safe_point(ff, safe_bo_run):
    order = np.array(safe_bo_run.order)
    assert (ff.T_true[order] <= T_LIMIT).all()
    f_star = ff.f_true[ff.safe_true].max()
    assert f_star - ff.f_true[order].max() < 1.0      # within 1 % efficiency


def test_unsafe_bo_does_trip_for_contrast(ff_small):
    h = st.bo_unsafe(ff_small, 20, np.random.default_rng(0), beta=2.5, seed=0)
    assert (ff_small.T_true[h.order] > T_LIMIT).sum() > 0

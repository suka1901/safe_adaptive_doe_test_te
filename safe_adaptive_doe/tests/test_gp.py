"""The GP wrapper must fit, quantify uncertainty sensibly, and strip measurement noise."""
from __future__ import annotations

import numpy as np
import pytest

from gp import GP

NOISE = 0.05


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, NOISE, len(X))
    return GP(noise_guess=1e-3).fit(X, y), X


def test_mean_follows_the_true_function(fitted):
    gp, _ = fitted
    Xt = np.random.default_rng(1).random((200, 2)) * 0.8 + 0.1     # inside the data cloud
    mu, _ = gp.predict(Xt)
    truth = np.sin(4 * Xt[:, 0]) + Xt[:, 1] ** 2
    assert np.sqrt(np.mean((mu - truth) ** 2)) < 0.1


def test_uncertainty_grows_away_from_the_data(fitted):
    gp, X = fitted
    _, sd_near = gp.predict(X[:5])
    _, sd_far = gp.predict(np.array([[3.0, 3.0]]))
    assert sd_far[0] > 5 * sd_near.max()


def test_learned_noise_is_close_to_the_true_noise(fitted):
    gp, _ = fitted
    assert gp.noise_sd == pytest.approx(NOISE, rel=0.5)


def test_predict_returns_latent_std_without_measurement_noise(fitted):
    """Total predictive variance = latent variance + noise variance."""
    gp, X = fitted
    _, sd_latent = gp.predict(X)
    _, sd_total = gp.model.predict(X, return_std=True)
    np.testing.assert_allclose(sd_total ** 2, sd_latent ** 2 + gp.noise_sd ** 2, rtol=1e-6)
    assert (sd_latent < sd_total).all()


def test_shapes_and_positivity(fitted):
    gp, _ = fitted
    mu, sd = gp.predict(np.zeros((7, 2)))
    assert mu.shape == sd.shape == (7,)
    assert (sd > 0).all()

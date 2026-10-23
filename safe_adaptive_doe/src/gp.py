"""Gaussian-process surrogate (scikit-learn) used for both efficiency and temperature."""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Sum, WhiteKernel


class GP:
    """Matern-5/2 GP with learned length scales (ARD) and a learned noise term."""

    def __init__(self, noise_guess: float, seed: int = 0):
        # One length scale per input (ARD). scikit-learn accepts an array here, but its
        # signature is annotated as float, hence the targeted ignore.
        matern = Matern(length_scale=np.array([0.25, 0.25]),  # pyright: ignore[reportArgumentType]
                        length_scale_bounds=(0.03, 1.0), nu=2.5)
        kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * matern
                  + WhiteKernel(noise_level=noise_guess, noise_level_bounds=(1e-6, 1e-1)))
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                              n_restarts_optimizer=2, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the *latent* function (measurement noise removed),
        which is what safety certification and uncertainty sampling should use."""
        mu, sd = cast(tuple[np.ndarray, np.ndarray], self.model.predict(X, return_std=True))
        noise_var = self._noise_level() * self._y_scale() ** 2
        sd_latent = np.sqrt(np.maximum(sd ** 2 - noise_var, 1e-12))
        return mu, sd_latent

    def _noise_level(self) -> float:
        """Fitted WhiteKernel noise level (in normalised-y units)."""
        fitted = cast(Sum, self.model.kernel_)          # ConstantKernel*Matern + WhiteKernel
        return float(cast(WhiteKernel, fitted.k2).noise_level)

    def _y_scale(self) -> float:
        """Standard deviation used by normalize_y to scale the targets."""
        return float(np.asarray(getattr(self.model, "_y_train_std")).ravel()[0])

    @property
    def noise_sd(self) -> float:
        return float(np.sqrt(self._noise_level()) * self._y_scale())

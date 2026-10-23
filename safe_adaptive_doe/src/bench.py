"""Synthetic engine test bench.

Operating space (normalised to [0, 1]^2):
    x[0] = engine speed  -> 1000 .. 6000 rpm
    x[1] = load          ->    0 .. 100 %
Measured outputs:
    efficiency  f(x)  in %   (the response we want to model / maximise)
    exhaust temperature T(x) in degC (safety constraint, limit 900 degC)

The safe-operating boundary is irregular and unknown to the algorithms.
A "full-factorial dataset" is a dense grid of noisy measurements, standing in
for an existing industrial dataset; algorithms query it sequentially (offline
emulation), exactly like the thesis plan. Querying a point above the limit
counts as a hardware trip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_LIMIT = 900.0          # degC
NOISE_EFF = 0.4          # % (1 sigma)
NOISE_TEMP = 4.0         # degC (1 sigma)


def efficiency(x: np.ndarray) -> np.ndarray:
    """True (noise-free) efficiency map in %. Smooth hill + steep ridge at high load."""
    s, l = x[..., 0], x[..., 1]
    hill = 14.0 * np.exp(-((s - 0.70) ** 2 / 0.07 + (l - 0.82) ** 2 / 0.09))   # peak lies beyond the limit
    ridge = 4.0 / (1.0 + np.exp(-(s - 0.35) / 0.02)) * l                     # steep step in speed
    tilt = 2.5 * np.sin(4.0 * l) * (1.0 - s)
    return 30.0 + hill + ridge + tilt


def temperature(x: np.ndarray) -> np.ndarray:
    """True (noise-free) exhaust temperature in degC."""
    s, l = x[..., 0], x[..., 1]
    return 520.0 + 300.0 * s + 400.0 * l ** 2 + 60.0 * np.sin(6.0 * s + 3.0 * l) * l


def to_units(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 1000.0 + 5000.0 * x[..., 0], 100.0 * x[..., 1]


@dataclass
class FullFactorial:
    X: np.ndarray            # (N, 2) grid points
    f_meas: np.ndarray       # noisy efficiency
    T_meas: np.ndarray       # noisy temperature
    f_true: np.ndarray
    T_true: np.ndarray
    n_side: int

    @property
    def safe_true(self) -> np.ndarray:
        return self.T_true <= T_LIMIT


def make_dataset(n_side: int = 41, seed: int = 0) -> FullFactorial:
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, 1.0, n_side)
    S, L = np.meshgrid(g, g, indexing="xy")
    X = np.column_stack([S.ravel(), L.ravel()])
    f_true, T_true = efficiency(X), temperature(X)
    return FullFactorial(X=X,
                         f_meas=f_true + rng.normal(0, NOISE_EFF, len(X)),
                         T_meas=T_true + rng.normal(0, NOISE_TEMP, len(X)),
                         f_true=f_true, T_true=T_true, n_side=n_side)

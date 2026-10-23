"""Sampling strategies that query the full-factorial pool one point at a time.

Every strategy returns a History: the order of queried pool indices. Querying a
point whose true temperature is above the limit is recorded as a trip
(constraint violation); on a real bench that point would not yield a usable
efficiency measurement, so it is never used to fit the efficiency model.

Strategies
  grid            nested uniform grids (classic static DoE), ignores safety
  random          uniformly random order, ignores safety
  safe_active     GP-based safe active learning for MODEL CALIBRATION
                  - cautious boundary tracing: only query points certified safe
                    (upper confidence bound of temperature below the limit)
                  - online density adaptation: prefer points where the efficiency
                    model is uncertain AND the response is steep; points near the
                    safe-set boundary get a bonus so the safe region keeps growing
  safe_bo         SafeOpt-style safe Bayesian OPTIMISATION: find the highest
                  efficiency without crossing the limit
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench import NOISE_EFF, NOISE_TEMP, T_LIMIT, FullFactorial
from gp import GP


@dataclass
class History:
    name: str
    order: list[int] = field(default_factory=list)


def seed_points(ff: FullFactorial, k: int = 4) -> list[int]:
    """A few conservative starting points in the low speed / low load corner (known to be safe)."""
    n = ff.n_side
    idx = [0, 2, 2 * n, 2 * n + 2][:k]
    return idx


def grid_order(ff: FullFactorial, budget: int, rng) -> History:
    """Nested uniform grids 2x2, 3x3, ... over the full domain (as an engineer would plan offline)."""
    n, order, seen = ff.n_side, [], set()
    for k in range(2, n + 1):
        ticks = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
        for j in ticks:
            for i in ticks:
                p = j * n + i
                if p not in seen:
                    seen.add(p)
                    order.append(p)
        if len(order) >= budget:
            break
    return History("Uniform grid", order[:budget])


def random_order(ff: FullFactorial, budget: int, rng) -> History:
    return History("Random sampling", list(rng.permutation(len(ff.X))[:budget]))


def _fit(ff: FullFactorial, order: list[int], seed: int):
    idx = np.array(order)
    ok = ff.T_meas[idx] <= T_LIMIT
    gT = GP(noise_guess=(NOISE_TEMP / 150) ** 2, seed=seed).fit(ff.X[idx], ff.T_meas[idx])
    gf = GP(noise_guess=(NOISE_EFF / 5) ** 2, seed=seed).fit(ff.X[idx[ok]], ff.f_meas[idx[ok]])
    return gf, gT


T_MARGIN = 10.0  # degC safety back-off below the hardware limit used for certification
REACH = 0.12   # max distance (normalised units, ~600 rpm / 12 % load) from a measured safe point


def safe_set(muT: np.ndarray, sdT: np.ndarray, beta: float, X: np.ndarray | None = None,
             safe_measured: np.ndarray | None = None) -> np.ndarray:
    """Certified-safe points: pessimistic (upper-bound) temperature below the limit AND
    within REACH of a point already measured safely, so the safe set only grows
    outwards from known-safe operation (as in SafeOpt) and never jumps to a far
    corner where the GP has simply fallen back to its prior mean."""
    ok = muT + beta * sdT <= T_LIMIT - T_MARGIN
    if X is not None and safe_measured is not None and len(safe_measured):
        d = np.min(np.linalg.norm(X[:, None, :] - X[safe_measured][None, :, :], axis=2), axis=1)
        ok &= d <= REACH
    return ok


def _measured_safe(ff: FullFactorial, order: list[int]) -> np.ndarray:
    idx = np.array(order)
    return idx[ff.T_meas[idx] <= T_LIMIT]


def safe_active(ff: FullFactorial, budget: int, rng, beta: float = 2.5,
                w_grad: float = 1.0, w_boundary: float = 0.5, seed: int = 0) -> History:
    n = ff.n_side
    order = seed_points(ff)
    while len(order) < budget:
        gf, gT = _fit(ff, order, seed)
        muT, sdT = gT.predict(ff.X)
        muf, sdf = gf.predict(ff.X)
        S = safe_set(muT, sdT, beta, ff.X, _measured_safe(ff, order))
        S[order] = False
        if not S.any():
            break
        # online density adaptation: uncertainty x local steepness of the efficiency model
        gy, gx = np.gradient(muf.reshape(n, n))
        grad = np.hypot(gx, gy).ravel()
        unc = sdf / sdf[S].max()
        steep = grad / (grad.max() + 1e-12)
        # cautious boundary tracing: bonus for safe points the temperature model is unsure about
        margin = (T_LIMIT - T_MARGIN - (muT + beta * sdT)) / (beta * sdT)       # 0 at the safe-set edge
        near_edge = np.exp(-margin) * (sdT / sdT[S].max())
        score = unc * (1.0 + w_grad * steep) + w_boundary * near_edge
        score[~S] = -np.inf
        order.append(int(np.argmax(score)))
    return History("Safe active learning (GP)", order)


def safe_bo(ff: FullFactorial, budget: int, rng, beta: float = 2.5, seed: int = 0) -> History:
    """SafeOpt-style: within the certified safe set, evaluate potential maximisers
    (upper bound above the best lower bound) and expanders (safe points at the edge
    that may enlarge the safe set), picking the most uncertain candidate."""
    order = seed_points(ff)
    while len(order) < budget:
        gf, gT = _fit(ff, order, seed)
        muT, sdT = gT.predict(ff.X)
        muf, sdf = gf.predict(ff.X)
        S = safe_set(muT, sdT, beta, ff.X, _measured_safe(ff, order))
        if not S.any():
            break
        ucb, lcb = muf + beta * sdf, muf - beta * sdf
        maximisers = S & (ucb >= lcb[S].max())
        margin = (T_LIMIT - T_MARGIN - (muT + beta * sdT)) / (beta * sdT)
        expanders = S & (margin < 0.5)
        width_f = np.where(maximisers, 2 * beta * sdf / (np.ptp(muf) + 1e-9), 0)
        width_T = np.where(expanders, 2 * beta * sdT / (np.ptp(muT) + 1e-9), 0)
        score = np.maximum(width_f, width_T)
        score[order] = -np.inf
        score[~S] = -np.inf
        if not np.isfinite(score).any():
            break
        order.append(int(np.argmax(score)))
    return History("Safe Bayesian optimisation (SafeOpt-style)", order)


def bo_unsafe(ff: FullFactorial, budget: int, rng, beta: float = 2.5, seed: int = 0) -> History:
    """Standard GP-UCB Bayesian optimisation that ignores the temperature limit (for contrast)."""
    order = seed_points(ff)
    while len(order) < budget:
        idx = np.array(order)
        ok = ff.T_meas[idx] <= T_LIMIT
        gf = GP(noise_guess=(NOISE_EFF / 5) ** 2, seed=seed).fit(ff.X[idx[ok]], ff.f_meas[idx[ok]])
        muf, sdf = gf.predict(ff.X)
        score = muf + beta * sdf
        score[order] = -np.inf
        order.append(int(np.argmax(score)))
    return History("Standard Bayesian optimisation (no safety)", order)

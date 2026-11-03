"""Acceptance rules and regression tolerances for pipeline results.

Used in two places:
  - src/make_baseline.py refuses to save a baseline that breaks a rule;
  - tests/test_pipeline_baseline.py checks every recorded case and every fresh rerun.

Each check returns a list of human-readable violations (empty list = pass), so a
failing test says exactly what went wrong.

`res` is one strategy's result for one seed, as written by run_experiments.py:
  DoE: {"n", "rmse", "coverage", "false_safe", "trips", "order"}
  BO:  {"n", "regret", "trips", "order"}
"""
from __future__ import annotations

import numpy as np

RMSE_TARGET = 0.5            # % efficiency points (same as run_experiments.RMSE_TARGET)
BO_TARGET = 0.5              # "within 0.5 % of the best safe point"

# ---- acceptance rules (absolute requirements) --------------------------------
SAFE_ACTIVE_MIN_COVERAGE = 0.90      # share of the true safe region certified at the end
SAFE_BO_MAX_TRIPS = 1                # per run (allows one noise-driven edge case)
SAFE_BO_MAX_BEST_REGRET = 1.5        # every run must get reasonably close
SAFE_BO_MIN_SHARE_REACHED = 0.7      # share of runs within BO_TARGET at some point

# ---- regression tolerances (fresh rerun vs recorded baseline) ----------------
RMSE_REL_TOL = 0.10                  # final RMSE may grow by 10 % ...
RMSE_ABS_TOL = 0.01                  # ... plus 0.01 points
COVERAGE_TOL = 0.03                  # certified share may drop by 3 points
REGRET_TOL = 0.25                    # best regret may grow by 0.25 points
MEDIAN_REACH_TOL = 5                 # median measurements-to-target may grow by one checkpoint

SAFE_STRATEGIES = ("safe_active", "safe_bo")
CHAOTIC = ("bo_unsafe",)             # unconstrained GP-UCB: tiny float differences change its path


def first_reach(values, ns, target):
    hit = np.where(np.array(values) <= target)[0]
    return ns[hit[0]] if len(hit) else None


# ---------------------------------------------------------------------------
# per case
# ---------------------------------------------------------------------------

def _common(res: dict, budget: int, n_points: int) -> list[str]:
    v = []
    order = res["order"]
    if len(order) != budget:
        v.append(f"{len(order)} measurements recorded, expected the budget of {budget}")
    if len(set(order)) != len(order):
        v.append("a point was measured twice")
    if min(order) < 0 or max(order) >= n_points:
        v.append("a measured index lies outside the dataset")
    if np.any(np.diff(res["trips"]) < 0):
        v.append("cumulative trips decrease")
    return v


def check_doe_case(strategy: str, res: dict, budget: int, n_points: int) -> list[str]:
    v = _common(res, budget, n_points)
    rmse, cov, fs, trips = (np.array(res[k]) for k in ("rmse", "coverage", "false_safe", "trips"))
    if not (np.isfinite(rmse).all() and (rmse > 0).all()):
        v.append("RMSE must be finite and positive")
    if ((cov < 0) | (cov > 1)).any():
        v.append("coverage outside [0, 1]")
    if fs.max() > 0:
        v.append(f"model certified {fs.max()} truly unsafe points as safe")
    if strategy == "safe_active":
        if trips.max() > 0:
            v.append(f"safe active learning exceeded the limit {trips.max()} times")
        if rmse[-1] > RMSE_TARGET:
            v.append(f"final RMSE {rmse[-1]:.3f} above target {RMSE_TARGET}")
        if cov[-1] < SAFE_ACTIVE_MIN_COVERAGE:
            v.append(f"only {cov[-1]:.0%} of the safe region certified (min {SAFE_ACTIVE_MIN_COVERAGE:.0%})")
    return v


def check_bo_case(strategy: str, res: dict, budget: int, n_points: int) -> list[str]:
    v = _common(res, budget, n_points)
    regret, trips = np.array(res["regret"]), np.array(res["trips"])
    if not np.isfinite(regret).all():
        v.append("regret must be finite")
    if strategy == "safe_bo":
        if trips[-1] > SAFE_BO_MAX_TRIPS:
            v.append(f"safe BO exceeded the limit {trips[-1]} times (max {SAFE_BO_MAX_TRIPS})")
        if regret.min() > SAFE_BO_MAX_BEST_REGRET:
            v.append(f"safe BO best regret {regret.min():.2f} above {SAFE_BO_MAX_BEST_REGRET}")
    if strategy == "bo_unsafe" and trips[-1] == 0:
        v.append("standard BO never tripped: the bench no longer tempts it beyond the limit")
    return v


# ---------------------------------------------------------------------------
# across seeds
# ---------------------------------------------------------------------------

def median_reach(runs: list[dict], strategy: str, metric: str, target: float) -> float | None:
    reach = [first_reach(r[strategy][metric], r[strategy]["n"], target) for r in runs]
    hit = [x for x in reach if x is not None]
    return float(np.median(hit)) if hit else None


def check_aggregate(doe: list[dict], bo: list[dict]) -> list[str]:
    v = []
    safe, grid = (median_reach(doe, m, "rmse", RMSE_TARGET) for m in ("safe_active", "grid"))
    if safe is None or (grid is not None and safe > grid):
        v.append(f"safe active learning needs more measurements than the grid (median {safe} vs {grid})")
    reached = sum(first_reach(r["safe_bo"]["regret"], r["safe_bo"]["n"], BO_TARGET) is not None for r in bo)
    if reached < SAFE_BO_MIN_SHARE_REACHED * len(bo):
        v.append(f"safe BO got within {BO_TARGET} % in only {reached}/{len(bo)} runs")
    return v


# ---------------------------------------------------------------------------
# regression: fresh rerun vs baseline
# ---------------------------------------------------------------------------

def compare_doe_case(strategy: str, new: dict, base: dict) -> list[str]:
    v = []
    if new["trips"][-1] > base["trips"][-1]:
        v.append(f"trips {base['trips'][-1]} -> {new['trips'][-1]}")
    if new["false_safe"][-1] > base["false_safe"][-1]:
        v.append(f"false-safe points {base['false_safe'][-1]} -> {new['false_safe'][-1]}")
    limit = base["rmse"][-1] * (1 + RMSE_REL_TOL) + RMSE_ABS_TOL
    if new["rmse"][-1] > limit:
        v.append(f"final RMSE {base['rmse'][-1]:.3f} -> {new['rmse'][-1]:.3f} (limit {limit:.3f})")
    if new["coverage"][-1] < base["coverage"][-1] - COVERAGE_TOL:
        v.append(f"certified share {base['coverage'][-1]:.1%} -> {new['coverage'][-1]:.1%}")
    return v


def compare_bo_case(strategy: str, new: dict, base: dict) -> list[str]:
    if strategy in CHAOTIC:
        return []
    v = []
    if strategy in SAFE_STRATEGIES and new["trips"][-1] > base["trips"][-1]:
        v.append(f"trips {base['trips'][-1]} -> {new['trips'][-1]}")
    if min(new["regret"]) > min(base["regret"]) + REGRET_TOL:
        v.append(f"best regret {min(base['regret']):.2f} -> {min(new['regret']):.2f}")
    return v


def compare_aggregate(new_doe, base_doe, new_bo, base_bo) -> list[str]:
    v = []
    a, b = median_reach(new_doe, "safe_active", "rmse", RMSE_TARGET), median_reach(base_doe, "safe_active", "rmse", RMSE_TARGET)
    if a is None or (b is not None and a > b + MEDIAN_REACH_TOL):
        v.append(f"safe active median measurements to target {b} -> {a}")
    count = lambda runs: sum(first_reach(r["safe_bo"]["regret"], r["safe_bo"]["n"], BO_TARGET) is not None for r in runs)
    if count(new_bo) < count(base_bo) - 1:
        v.append(f"safe BO runs within target {count(base_bo)} -> {count(new_bo)}")
    return v

"""Offline sequential emulation on full-factorial datasets.

Experiment 1 - model calibration (adaptive DoE):
    How many measurements does each strategy need to model efficiency over the
    safe operating region to a target accuracy, and how many limit violations
    (trips) does it cause on the way?
Experiment 2 - safe Bayesian optimisation:
    How fast does each strategy find the most efficient safe operating point,
    and how many trips does it cause?

Each experiment is repeated over several random seeds (new measurement noise
and, for random sampling, a new order). Results are written to results/.

Usage:  python src/run_experiments.py [--seeds 10] [--budget 100] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import strategies as st
from bench import T_LIMIT, make_dataset, to_units

RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKPOINTS = list(range(10, 101, 5))
BO_CHECKPOINTS = list(range(6, 61, 2))
RMSE_TARGET = 0.5          # % efficiency points, over the true safe region
BETA = 2.5                 # confidence multiplier for the safe set (~99.4 % one-sided)
DOE = {"grid": st.grid_order, "random": st.random_order, "safe_active": st.safe_active}
BO = {"grid": st.grid_order, "random": st.random_order, "bo_unsafe": st.bo_unsafe, "safe_bo": st.safe_bo}


def evaluate_model(ff, order, seed):
    gf, gT = st._fit(ff, order, seed)
    muf, _ = gf.predict(ff.X)
    muT, sdT = gT.predict(ff.X)
    safe = ff.safe_true
    rmse = float(np.sqrt(np.mean((muf[safe] - ff.f_true[safe]) ** 2)))
    certified = st.safe_set(muT, sdT, BETA, ff.X, st._measured_safe(ff, order))
    coverage = float((certified & safe).sum() / safe.sum())
    false_safe = int((certified & ~safe).sum())
    trips = int((ff.T_true[order] > T_LIMIT).sum())
    return rmse, coverage, false_safe, trips


GP_STRATEGIES = {"safe_active", "safe_bo", "bo_unsafe"}


def generate_orders(exp: str, seed: int, budget: int, ff=None) -> dict[str, st.History]:
    """Run every strategy of one experiment ("doe" or "bo") for one seed.

    The strategies share one random generator in a fixed order, so this function is the
    single place that defines how a seed maps to measurement sequences (the replay tests use it).
    """
    ff = make_dataset(seed=seed) if ff is None else ff
    table, offset = (DOE, 1000) if exp == "doe" else (BO, 2000)
    rng = np.random.default_rng(offset + seed)
    out = {}
    for key, fn in table.items():
        kw = {"beta": BETA, "seed": seed} if key in GP_STRATEGIES else {}
        out[key] = fn(ff, budget, rng, **kw)
    return out


def bo_checkpoint(ff, order, n: int, f_star: float) -> tuple[float, int]:
    """Regret of the recommended point (best measured safe point) and trips after n measurements."""
    idx = np.array(order[:n])
    ok = idx[ff.T_meas[idx] <= T_LIMIT]                     # usable measurements
    best = ok[np.argmax(ff.f_meas[ok])]                       # recommend best measured safe point
    return float(f_star - ff.f_true[best]), int((ff.T_true[idx] > T_LIMIT).sum())


def run_doe_seed(seed: int, budget: int) -> dict:
    ff = make_dataset(seed=seed)
    out = {}
    for key, h in generate_orders("doe", seed, budget, ff).items():
        res = {"n": [], "rmse": [], "coverage": [], "false_safe": [], "trips": []}
        for n in CHECKPOINTS:
            if n > len(h.order):
                break
            r, c, fs, t = evaluate_model(ff, h.order[:n], seed)
            for k, v in zip(("n", "rmse", "coverage", "false_safe", "trips"), (n, r, c, fs, t)):
                res[k].append(v)
        res["name"], res["order"] = h.name, [int(i) for i in h.order]
        out[key] = res
    return out


def run_bo_seed(seed: int, budget: int) -> dict:
    ff = make_dataset(seed=seed)
    f_star = float(ff.f_true[ff.safe_true].max())
    out = {}
    for key, h in generate_orders("bo", seed, budget, ff).items():
        res = {"n": [], "regret": [], "trips": []}
        for n in BO_CHECKPOINTS:
            if n > len(h.order):
                break
            regret, trips = bo_checkpoint(ff, h.order, n, f_star)
            res["n"].append(n)
            res["regret"].append(regret)
            res["trips"].append(trips)
        res["name"], res["order"] = h.name, [int(i) for i in h.order]
        out[key] = res
    return out


def stack(runs, method, metric):
    L = min(len(r[method][metric]) for r in runs)
    return np.array([r[method][metric][:L] for r in runs]), runs[0][method]["n"][:L]


def first_reach(values, ns, target):
    hit = np.where(np.array(values) <= target)[0]
    return ns[hit[0]] if len(hit) else None


COLORS = {"grid": "#7f7f7f", "random": "#bcbd22", "safe_active": "#1f6feb",
          "bo_unsafe": "#d9480f", "safe_bo": "#1f6feb"}


def plot_doe(runs, ff0, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in DOE:
        for k, (metric, a) in enumerate([("rmse", ax[0]), ("trips", ax[1]), ("coverage", ax[2])]):
            Y, ns = stack(runs, m, metric)
            mu, sd = Y.mean(0), Y.std(0)
            a.plot(ns, mu, color=COLORS[m], lw=2, label=runs[0][m]["name"])
            a.fill_between(ns, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
    ax[0].axhline(RMSE_TARGET, color="k", ls=":", lw=1)
    ax[0].set(title="Model error over the safe region", xlabel="measurements", ylabel="RMSE (% efficiency)")
    ax[0].set_yscale("log")
    ax[1].set(title="Limit violations (trips)", xlabel="measurements", ylabel="cumulative trips")
    ax[2].set(title="Safe region certified by the model", xlabel="measurements", ylabel="share of true safe region")
    ax[0].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_maps(run0, ff0, n_show, path):
    n = ff0.n_side
    s, l = to_units(ff0.X)
    S, L = s.reshape(n, n), l.reshape(n, n)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    levels = np.linspace(ff0.f_true.min(), ff0.f_true.max(), 19)
    c = None
    for a, m in zip(ax, ["grid", "safe_active"]):
        c = a.contourf(S, L, ff0.f_true.reshape(n, n), levels=levels, cmap="viridis")
        a.contour(S, L, ff0.T_true.reshape(n, n), [T_LIMIT], colors="red", linewidths=2)
        idx = np.array(run0[m]["order"][:n_show])
        bad = ff0.T_true[idx] > T_LIMIT
        ps, pl = to_units(ff0.X[idx])
        a.scatter(ps[~bad], pl[~bad], c="white", edgecolors="k", s=28, label="measurement")
        a.scatter(ps[bad], pl[bad], marker="x", c="red", s=60, linewidths=2, label="trip (over limit)")
        a.set(title=f"{run0[m]['name']} - first {n_show} points", xlabel="engine speed (rpm)")
        a.legend(loc="lower right", fontsize=8)
    ax[0].set_ylabel("load (%)")
    assert c is not None
    fig.colorbar(c, ax=ax, label="true efficiency (%)  -  red line: 900 degC limit")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_bo(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in BO:
        for metric, a in [("regret", ax[0]), ("trips", ax[1])]:
            Y, ns = stack(runs, m, metric)
            mu = Y.mean(0)
            a.plot(ns, mu, color=COLORS[m], lw=2, ls="--" if m == "bo_unsafe" else "-", label=runs[0][m]["name"])
    ax[0].set(title="Gap to best safe efficiency", xlabel="measurements", ylabel="regret (% efficiency)")
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[1].set(title="Limit violations (trips)", xlabel="measurements", ylabel="cumulative trips")
    ax[0].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    seeds = list(range(args.seeds))

    with ProcessPoolExecutor(args.workers) as ex:
        doe = list(ex.map(run_doe_seed, seeds, [args.budget] * len(seeds)))
        bo = list(ex.map(run_bo_seed, seeds, [60] * len(seeds)))
    (RESULTS / "raw_results.json").write_text(json.dumps({"doe": doe, "bo": bo}))

    ff0 = make_dataset(seed=0)
    plot_doe(doe, ff0, RESULTS / "doe_curves.png")
    plot_maps(doe[0], ff0, 40, RESULTS / "sample_maps.png")
    plot_bo(bo, RESULTS / "safe_bo.png")

    lines = ["# Results", "", f"{len(seeds)} seeds, offline emulation on a 41x41 full-factorial dataset "
             f"(1681 points, {ff0.safe_true.mean():.0%} safe). Confidence beta = {BETA}.", "",
             "## Experiment 1 - model calibration", "",
             f"| Strategy | Measurements to reach RMSE <= {RMSE_TARGET} % (median) | RMSE at 40 | RMSE at 100 | Trips at 40 | Trips at 100 | Safe region certified at 100 |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for m in DOE:
        R, ns = stack(doe, m, "rmse")
        T, _ = stack(doe, m, "trips")
        C, _ = stack(doe, m, "coverage")
        reach = [first_reach(r, ns, RMSE_TARGET) for r in R]
        reached = [x for x in reach if x is not None]
        med = f"{int(np.median(reached))} ({len(reached)}/{len(R)} runs)" if reached else f"not reached in {ns[-1]}"
        i40 = ns.index(40) if 40 in ns else len(ns) - 1     # short runs: use the last checkpoint
        summary[m] = {"reach": reached, "rmse40": R[:, i40].mean(), "trips100": T[:, -1].mean()}
        lines.append(f"| {doe[0][m]['name']} | {med} | {R[:, i40].mean():.2f} | {R[:, -1].mean():.2f} "
                     f"| {T[:, i40].mean():.1f} | {T[:, -1].mean():.1f} | {C[:, -1].mean():.0%} |")
    lines += ["", "## Experiment 2 - safe Bayesian optimisation (find best safe efficiency)", "",
              "| Strategy | Measurements to get within 0.5 % of best safe point (median) | Regret at 20 | Regret at 60 | Trips at 60 |",
              "|---|---|---|---|---|"]
    for m in BO:
        R, ns = stack(bo, m, "regret")
        T, _ = stack(bo, m, "trips")
        reach = [first_reach(r, ns, 0.5) for r in R]
        reached = [x for x in reach if x is not None]
        med = f"{int(np.median(reached))} ({len(reached)}/{len(R)} runs)" if reached else "not reached"
        lines.append(f"| {bo[0][m]['name']} | {med} | {R[:, ns.index(20)].mean():.2f} | {R[:, -1].mean():.2f} | {T[:, -1].mean():.1f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nTo turn these results into the regression test baseline: python src/make_baseline.py")


if __name__ == "__main__":
    main()

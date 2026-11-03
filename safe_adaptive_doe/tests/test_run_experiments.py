"""Evaluation helpers and an end-to-end smoke test of the experiment pipeline."""
from __future__ import annotations

import sys

import numpy as np
import pytest

import run_experiments as rx
import strategies as st


def test_first_reach_returns_first_checkpoint_below_target():
    assert rx.first_reach([3.0, 1.0, 0.4, 0.6, 0.2], [10, 15, 20, 25, 30], 0.5) == 20
    assert rx.first_reach([3.0, 1.0], [10, 15], 0.5) is None


def test_stack_truncates_to_shortest_run():
    runs = [{"m": {"rmse": [1, 2, 3], "n": [10, 15, 20]}},
            {"m": {"rmse": [4, 5], "n": [10, 15]}}]
    Y, ns = rx.stack(runs, "m", "rmse")
    assert Y.shape == (2, 2) and ns == [10, 15]


def test_evaluate_model_returns_valid_metrics(ff):
    order = st.grid_order(ff, 25, np.random.default_rng(0)).order
    rmse, coverage, false_safe, trips = rx.evaluate_model(ff, order, seed=0)
    assert 0 < rmse < 10
    assert 0.0 <= coverage <= 1.0
    assert false_safe >= 0
    assert trips == int((ff.T_true[order] > rx.T_LIMIT).sum())


def test_doe_seed_runs_all_strategies(monkeypatch):
    monkeypatch.setattr(rx, "CHECKPOINTS", [10, 15])
    out = rx.run_doe_seed(seed=0, budget=15)
    assert set(out) == {"grid", "random", "safe_active"}
    assert out["safe_active"]["trips"][-1] == 0
    for res in out.values():
        assert res["n"] == [10, 15]


@pytest.mark.slow
def test_full_pipeline_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--seeds", "2", "--budget", "20", "--workers", "1"])
    rx.main()
    for name in ("summary.md", "raw_results.json", "doe_curves.png", "sample_maps.png", "safe_bo.png"):
        assert (tmp_path / name).exists(), name
    summary = (tmp_path / "summary.md").read_text()
    assert "Safe active learning (GP)" in summary and "SafeOpt" in summary

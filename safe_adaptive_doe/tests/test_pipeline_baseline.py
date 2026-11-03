"""Tests generated from the pipeline's own results (tests/baseline/pipeline_baseline.json).

One test per strategy; each covers every recorded seed and names the seeds that fail.
More seeds in the pipeline means more coverage, without more test entries.

1. Sync       - the baseline matches results/raw_results.json and the pipeline settings.
2. Replay     - for every seed, each strategy must choose the same first measurements again.
3. Regression - (slow) every seed is rerun in full; the result must meet the acceptance
                rules in src/acceptance.py and not be worse than the baseline beyond the
                tolerances defined there.

The recorded results themselves are checked against the acceptance rules by
src/make_baseline.py, which refuses to save a baseline that breaks them.

Update the baseline after a deliberate change:  python src/make_baseline.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pytest

import acceptance as acc
import make_baseline
import run_experiments as rx
from bench import make_dataset

BASELINE = Path(__file__).parent / "baseline" / "pipeline_baseline.json"
if not BASELINE.exists():
    pytest.skip("no baseline yet - run: python src/make_baseline.py", allow_module_level=True)

B = json.loads(BASELINE.read_text())
META = B["meta"]
SEEDS = list(range(META["seeds"]))
DOE_STRATEGIES = list(B["doe"][0])
BO_STRATEGIES = list(B["bo"][0])
REPLAY_BUDGET = 15                     # first measurements re-chosen in the replay
STALE = "the baseline is out of date - run: python src/make_baseline.py"


@lru_cache(maxsize=None)
def n_points() -> int:
    return len(make_dataset(seed=0).X)


@lru_cache(maxsize=None)
def replayed_orders(exp: str, seed: int) -> dict[str, list[int]]:
    hist = rx.generate_orders(exp, seed, REPLAY_BUDGET)
    return {k: [int(i) for i in h.order] for k, h in hist.items()}


def report(failures: dict[int, list[str]]) -> str:
    return "".join(f"\n  seed {s}: " + "; ".join(msgs) for s, msgs in failures.items())


# ---------------------------------------------------------------------------
# 1. sync
# ---------------------------------------------------------------------------

def test_baseline_matches_latest_pipeline_results():
    results = make_baseline.DEFAULT_RESULTS
    if not results.exists():
        pytest.skip("results/raw_results.json not present")
    assert make_baseline.fingerprint(json.loads(results.read_text())) == META["source_fingerprint"], STALE


def test_pipeline_settings_match_the_baseline():
    assert rx.CHECKPOINTS[:len(META["doe_checkpoints"])] == META["doe_checkpoints"], STALE
    assert rx.BO_CHECKPOINTS[:len(META["bo_checkpoints"])] == META["bo_checkpoints"], STALE


# ---------------------------------------------------------------------------
# 2. replay: same seed in -> same measurement sequence out
#    (standard BO without a safety model is left out: its path reacts to
#     floating-point differences between machines)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp, strategy",
                         [("doe", m) for m in DOE_STRATEGIES] +
                         [("bo", m) for m in BO_STRATEGIES if m not in acc.CHAOTIC])
def test_replay_chooses_the_recorded_points(exp, strategy):
    failures = {}
    for seed in SEEDS:
        got = replayed_orders(exp, seed)[strategy]
        want = B[exp][seed][strategy]["order"][:REPLAY_BUDGET]
        if got != want:
            k = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            failures[seed] = [f"differs from measurement {k + 1}"]
    assert not failures, report(failures)


# ---------------------------------------------------------------------------
# 3. regression: full rerun vs baseline (slow, ~10-15 min)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rerun():
    with ProcessPoolExecutor() as ex:
        doe = list(ex.map(rx.run_doe_seed, SEEDS, [META["doe_budget"]] * len(SEEDS)))
        bo = list(ex.map(rx.run_bo_seed, SEEDS, [META["bo_budget"]] * len(SEEDS)))
    return {"doe": doe, "bo": bo}


@pytest.mark.slow
@pytest.mark.parametrize("strategy", DOE_STRATEGIES)
def test_rerun_doe(rerun, strategy):
    failures = {}
    for seed in SEEDS:
        new, base = rerun["doe"][seed][strategy], B["doe"][seed][strategy]
        msgs = acc.check_doe_case(strategy, new, META["doe_budget"], n_points())
        msgs += acc.compare_doe_case(strategy, new, base)
        if msgs:
            failures[seed] = msgs
    assert not failures, report(failures)


@pytest.mark.slow
@pytest.mark.parametrize("strategy", BO_STRATEGIES)
def test_rerun_bo(rerun, strategy):
    failures = {}
    for seed in SEEDS:
        new, base = rerun["bo"][seed][strategy], B["bo"][seed][strategy]
        msgs = acc.check_bo_case(strategy, new, META["bo_budget"], n_points())
        msgs += acc.compare_bo_case(strategy, new, base)
        if msgs:
            failures[seed] = msgs
    assert not failures, report(failures)


@pytest.mark.slow
def test_rerun_across_seeds(rerun):
    msgs = acc.check_aggregate(rerun["doe"], rerun["bo"])
    msgs += acc.compare_aggregate(rerun["doe"], B["doe"], rerun["bo"], B["bo"])
    assert not msgs, "\n  - " + "\n  - ".join(msgs)

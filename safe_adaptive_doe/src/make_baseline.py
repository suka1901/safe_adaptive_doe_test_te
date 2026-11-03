"""Turn the pipeline output into the test baseline.

Reads results/raw_results.json (written by run_experiments.py), checks every
recorded case against the acceptance rules in acceptance.py, prints what changed
since the previous baseline, and writes tests/baseline/pipeline_baseline.json.
The tests in tests/test_pipeline_baseline.py are generated from that file:
one test case per seed and strategy.

Workflow after a deliberate change to the method:
    python src/run_experiments.py      # new results
    python src/make_baseline.py        # check them and update the baseline
    git diff tests/baseline            # review, then commit

Usage:  python src/make_baseline.py [--results PATH] [--out PATH] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

import acceptance as acc
from bench import make_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "raw_results.json"
DEFAULT_OUT = ROOT / "tests" / "baseline" / "pipeline_baseline.json"


def fingerprint(raw: dict) -> str:
    """Hash of the results content (independent of file formatting and line endings)."""
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def collect_violations(raw: dict) -> list[str]:
    n_points = len(make_dataset(seed=0).X)
    out = []
    for exp, check in (("doe", acc.check_doe_case), ("bo", acc.check_bo_case)):
        for seed, run in enumerate(raw[exp]):
            for strategy, res in run.items():
                budget = len(res["order"])
                out += [f"{exp} seed {seed} {strategy}: {msg}" for msg in check(strategy, res, budget, n_points)]
    out += [f"aggregate: {msg}" for msg in acc.check_aggregate(raw["doe"], raw["bo"])]
    return out


def summarise(raw: dict) -> dict:
    doe, bo = raw["doe"], raw["bo"]
    return {
        "safe_active median to RMSE target": acc.median_reach(doe, "safe_active", "rmse", acc.RMSE_TARGET),
        "grid median to RMSE target": acc.median_reach(doe, "grid", "rmse", acc.RMSE_TARGET),
        "safe_active trips (total)": int(sum(r["safe_active"]["trips"][-1] for r in doe)),
        "safe_active mean final RMSE": round(float(np.mean([r["safe_active"]["rmse"][-1] for r in doe])), 3),
        "safe_bo runs within target": int(sum(acc.first_reach(r["safe_bo"]["regret"], r["safe_bo"]["n"],
                                                              acc.BO_TARGET) is not None for r in bo)),
        "safe_bo trips (total)": int(sum(r["safe_bo"]["trips"][-1] for r in bo)),
    }


def build(raw: dict, source: Path) -> dict:
    first_doe, first_bo = raw["doe"][0], raw["bo"][0]
    return {
        "meta": {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source": source.name,
            "source_fingerprint": fingerprint(raw),
            "seeds": len(raw["doe"]),
            "doe_budget": len(first_doe["safe_active"]["order"]),
            "bo_budget": len(first_bo["safe_bo"]["order"]),
            "doe_checkpoints": first_doe["safe_active"]["n"],
            "bo_checkpoints": first_bo["safe_bo"]["n"],
            "written_with": {"python": platform.python_version(), "numpy": np.__version__,
                             "scikit_learn": sklearn.__version__},
        },
        "summary": summarise(raw),
        "doe": raw["doe"],
        "bo": raw["bo"],
    }


def dump(obj: dict) -> str:
    """JSON with one line per list of numbers, so `git diff` shows changes case by case."""
    text = json.dumps(obj, indent=1)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="save even if acceptance rules are broken")
    args = ap.parse_args()

    raw = json.loads(args.results.read_text())
    violations = collect_violations(raw)
    if violations:
        print(f"{len(violations)} acceptance rule(s) broken:")
        print("\n".join(f"  - {v}" for v in violations))
        if not args.force:
            sys.exit("Baseline NOT updated. Fix the method, or use --force if the change is intended.")

    new = build(raw, args.results)
    if args.out.exists():
        old = json.loads(args.out.read_text())
        if old["meta"]["source_fingerprint"] == new["meta"]["source_fingerprint"]:
            print("Baseline already matches these results - nothing to do.")
            return
        print("Changes since the previous baseline:")
        for k, v in new["summary"].items():
            was = old.get("summary", {}).get(k)
            print(f"  {k}: {was} -> {v}" + ("" if was != v else "  (same)"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump(new))
    n_cases = sum(len(r) for r in raw["doe"]) + sum(len(r) for r in raw["bo"])
    print(f"Wrote {args.out} - {n_cases} recorded cases from {len(raw['doe'])} seeds. "
          "Review with git diff, then commit.")


if __name__ == "__main__":
    main()

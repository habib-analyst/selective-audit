"""Run the selective-audit benchmark and write results/results.json.

Design: for each regime x seed, one stream is generated and shared across all
policies and budgets (common random numbers — paired comparison). Policies
get independent decision RNGs. Reports detections per policy x regime x
budget, oracle detections, and regret = oracle - policy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_audit import POLICIES, generate_stream, oracle_detections, run_policy
from selective_audit.engine import oracle_expected
from selective_audit.streams import REGIMES

T = 20_000
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [0.01, 0.05, 0.10]


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "results" / "results.json"
    results: dict = {
        "config": {"T": T, "seeds": SEEDS, "budgets": BUDGETS, "n_segments": 8},
        "runs": [],
        "summary": {},
    }
    t0 = time.time()
    for regime in REGIMES:
        for seed in SEEDS:
            stream = generate_stream(regime, T=T, seed=seed)
            oracle = {b: oracle_detections(stream, b) for b in BUDGETS}
            oracle_exp = {b: oracle_expected(stream, b) for b in BUDGETS}
            for b in BUDGETS:
                for name, cls in POLICIES.items():
                    policy = cls()
                    res = run_policy(stream, policy, b, rng_seed=10_000 + seed)
                    row = {
                        "regime": regime,
                        "seed": seed,
                        "budget_frac": b,
                        "policy": name,
                        "detections": res["detections"],
                        "audits_used": res["audits_used"],
                        "budget": res["budget"],
                        "oracle_detections": oracle[b],
                        "oracle_expected": oracle_exp[b],
                        "regret": oracle_exp[b] - res["detections"],
                        "total_violations": res["total_violations"],
                    }
                    if isinstance(policy, POLICIES["drift_ucb"]):
                        row["cp_resets"] = policy.resets
                    results["runs"].append(row)
            print(f"{regime} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)

    # summary: mean +/- std over seeds per (regime, budget, policy)
    for regime in REGIMES:
        for b in BUDGETS:
            for name in POLICIES:
                dets = [
                    r["detections"]
                    for r in results["runs"]
                    if r["regime"] == regime and r["budget_frac"] == b and r["policy"] == name
                ]
                regs = [
                    r["regret"]
                    for r in results["runs"]
                    if r["regime"] == regime and r["budget_frac"] == b and r["policy"] == name
                ]
                key = f"{regime}|{b}|{name}"
                results["summary"][key] = {
                    "detections_mean": float(np.mean(dets)),
                    "detections_std": float(np.std(dets)),
                    "regret_mean": float(np.mean(regs)),
                    "regret_std": float(np.std(regs)),
                }
            ora = [
                r["oracle_expected"]
                for r in results["runs"]
                if r["regime"] == regime and r["budget_frac"] == b and r["policy"] == "uniform"
            ]
            results["summary"][f"{regime}|{b}|oracle"] = {
                "detections_mean": float(np.mean(ora)),
                "detections_std": float(np.std(ora)),
            }

    results["wall_seconds"] = time.time() - t0
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path} in {results['wall_seconds']:.0f}s")

    # console headline table
    for regime in REGIMES:
        for b in BUDGETS:
            cells = []
            for name in list(POLICIES) + ["oracle"]:
                s = results["summary"][f"{regime}|{b}|{name}"]
                cells.append(f"{name}={s['detections_mean']:.0f}±{s['detections_std']:.0f}")
            print(f"{regime:7s} B={b:4.0%}  " + "  ".join(cells))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RQ1 robustness: is the fingerprint-only >> return-only asymmetry (Finding 2,
Section 5.2) just an artefact of the two detectors running at different
false-positive rates?

The fingerprint decision runs at a measured 10-15% empirical FPR (the per-env
Random-policy floor), while the return z-test runs at nominal 5% (|z|>1.96). A
skeptic could argue the fingerprint "finds more" simply because it is tuned to
fire more easily. This script tests that by re-running the RETURN detector at a
threshold matched to the fingerprint's FPR and re-counting fp-only vs return-only.

Method:
  1. On the stationary Random policy, measure the return test's own empirical
     FPR at |z|>1.96 (every flagged pair is a false positive) and find the
     threshold zc_match whose FPR equals the fingerprint's per-env floor.
  2. On the learners, recount fp-only and return-only pairs at the nominal
     (|z|>1.96) and matched (|z|>zc_match) return thresholds.

Findings this reproduces (paper footnote):
  - The return test's empirical FPR on Random is environment-dependent
    (~0.4% MountainCar, ~0.7% FrozenLake, ~26% Taxi) -- it is NOT uniformly the
    stricter detector (in Taxi it is more permissive than the fingerprint).
  - FrozenLake (the cleanly matchable case): matching lifts return-only 12->27
    while fingerprint-only still dominates ~20:1.
  - MountainCar: the Random-policy return is essentially constant, so no
    threshold reproduces a 10% FPR (zc_match ~ 0); the small return-only count
    reflects the absence of return movement, not detector stringency.
  => The one-directional information gain survives FPR-matching; it is not a
     thresholding artefact. The matched test is a stress test, NOT a detector we
     would deploy (a 15%-FPR return monitor is strictly worse in practice).

Both detectors match scripts/two_detector_decisions.py.

USAGE:
    python scripts/rq1_fpr_match.py
    python scripts/rq1_fpr_match.py --results_root results --mode standard
"""
import argparse
import glob
import json
import math
import os

import numpy as np

Z_CRIT = 1.9599639845400545  # two-sided p<0.05
EVO_METRICS = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]
ENV_N = {"Taxi": 1000, "FrozenLake": 500, "MountainCar": 500}
ENV_DIR = {"FrozenLake": "frozenlake8x8", "MountainCar": "mountain_car", "Taxi": "taxi"}
# Paper's measured fingerprint false-positive floor per environment (Sec null_fpr).
FP_FLOOR = {"FrozenLake": 0.15, "MountainCar": 0.10, "Taxi": 0.14}
ALGOS = ["q_learning", "sarsa", "dqn", "ppo"]
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5", "seed4242"]
EXCLUDE = {("Taxi", "ppo")}
# Non-standard random variants to exclude when estimating the STATIONARY floor.
NON_STATIONARY = ("reward_hacking", "perpetual", "limited")


def _arr(d, k):
    return np.array([np.nan if v is None else float(v) for v in d[k]], dtype=float)


def has_null(d):
    return isinstance(d.get("null_std_topological_shift"), list) and isinstance(d.get("zmax_p95"), list)


def excess_z(d):
    """Per-pair max-statistic minus family-wise detection threshold (>0 == active)."""
    zs = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in EVO_METRICS:
            sd = _arr(d, "null_std_" + k)
            z = (_arr(d, k + "_raw") - _arr(d, "null_mean_" + k)) / sd
            z[sd == 0] = np.nan  # skip degenerate metrics (matches pipeline 'continue')
            zs.append(z)
    return np.nanmax(np.vstack(zs), axis=0) - _arr(d, "zmax_p95")


def return_absz(d, n):
    """Per-pair |z| of the two-sample z-test on the mean return."""
    mr, sr = _arr(d, "mean_return"), _arr(d, "std_return")
    z = np.full(len(mr) - 1, np.nan)
    for i in range(len(mr) - 1):
        se2 = sr[i] ** 2 / n + sr[i + 1] ** 2 / n
        diff = mr[i + 1] - mr[i]
        z[i] = (0.0 if diff == 0 else math.copysign(1e9, diff)) if se2 <= 0 else diff / math.sqrt(se2)
    return np.abs(z)


def random_return_z(root, env):
    """|z| of the return test over the stationary Random policy (standard only)."""
    out = []
    for f in glob.glob(os.path.join(root, ENV_DIR[env], "random", "**", "*_metrics.json"), recursive=True):
        if any(t in f for t in NON_STATIONARY):
            continue
        out.append(return_absz(json.load(open(f)), ENV_N[env]))
    z = np.concatenate(out)
    return z[~np.isnan(z)]


def learner_runs(root, env, mode):
    for a in ALGOS:
        if (env, a) in EXCLUDE:
            continue
        for s in SEEDS:
            fs = glob.glob(os.path.join(root, ENV_DIR[env], a, mode, s, "*_metrics.json"))
            if fs:
                d = json.load(open(fs[0]))
                if has_null(d):
                    yield d


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="RQ1 FPR-matching robustness (Finding 2 footnote).")
    ap.add_argument("--results_root", default=os.path.join(here, "results"))
    ap.add_argument("--mode", default="standard")
    args = ap.parse_args()

    print(f"{'env':12} {'ret_FPR@1.96':>12} {'fp_floor':>9} {'zc_match':>9}")
    zc_match = {}
    for env in ENV_DIR:
        rz = random_return_z(args.results_root, env)
        nominal = float((rz > Z_CRIT).mean())
        zc = float(np.quantile(rz, 1 - FP_FLOOR[env]))
        zc_match[env] = zc
        deg = "  (~0: Random return ~constant, match ill-posed)" if zc < 1e-6 else ""
        print(f"{env:12} {nominal:12.3f} {FP_FLOOR[env]:9.2f} {zc:9.3f}{deg}")

    print(f"\n{'env':12} {'nominal fp-only/ret-only':>26} {'matched fp-only/ret-only':>26}")
    tot = [0, 0, 0, 0]
    for env in ENV_DIR:
        nfo = nro = mfo = mro = 0
        for d in learner_runs(args.results_root, env, args.mode):
            ex = excess_z(d)
            fp = np.where(np.isnan(ex), False, ex > 0)
            z = return_absz(d, ENV_N[env])
            cn, cm = z > Z_CRIT, z > zc_match[env]
            nfo += int((fp & ~cn).sum()); nro += int((~fp & cn).sum())
            mfo += int((fp & ~cm).sum()); mro += int((~fp & cm).sum())
        tot[0] += nfo; tot[1] += nro; tot[2] += mfo; tot[3] += mro
        ratio = f"~{mfo / mro:.0f}:1" if mro else "inf"
        print(f"{env:12} {f'{nfo} / {nro}':>26} {f'{mfo} / {mro}  ({ratio})':>26}")
    print(f"{'TOTAL':12} {f'{tot[0]} / {tot[1]}':>26} {f'{tot[2]} / {tot[3]}':>26}")
    print("\nNote: MountainCar's matched threshold is ~0 (Random return is constant), so its"
          "\nmatched counts are degenerate; FrozenLake is the cleanly matchable case (~20:1).")


if __name__ == "__main__":
    main()

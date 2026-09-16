#!/usr/bin/env python3
"""R4: confidence intervals to match RQ3's rigor.
  - RQ1 Finding 1 per-environment fingerprint-active rate: cluster bootstrap over runs.
  - RQ2 negative-control rate (11/12): Wilson score interval.
(The dose-response rho CI is in rq1_dose_response.py; the forecast AUC CI is in rq1_forecast_redundancy.py.)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_refresh as R

ROOT = r"C:\Users\ondra\Documents\metric_results"


def wilson(k, n, z=1.9599639845400545):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


print("=== RQ1 Finding 1: per-env fingerprint-active rate, cluster bootstrap over runs ===")
rng = np.random.default_rng(0)
for env in R.ENV_DIR:
    runs = []
    for algo in R.ALGOS:
        if (env, algo) in R.EXCLUDE:
            continue
        for _s, (fp, _ret) in R.learner_seeds(ROOT, env, algo):
            runs.append((int(fp.sum()), len(fp)))
    A = sum(a for a, _ in runs); T = sum(t for _, t in runs); rate = A / T
    boots = []
    for _ in range(5000):
        idx = rng.integers(0, len(runs), len(runs))
        a = sum(runs[i][0] for i in idx); t = sum(runs[i][1] for i in idx)
        boots.append(a / t)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {env:12} active {rate*100:.0f}% [{lo*100:.0f}%, {hi*100:.0f}%]  (n_runs={len(runs)})")

print("\n=== RQ2 negative controls: Wilson 95% CI ===")
lo, hi = wilson(11, 12)
print(f"  11/12 correctly unseparated = {11/12*100:.0f}%  Wilson 95% CI [{lo*100:.0f}%, {hi*100:.0f}%]")

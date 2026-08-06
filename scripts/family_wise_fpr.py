"""
Family-wise (max-statistic) fingerprint-active decision, per checkpoint pair.

The permutation pipeline (experiments/sequential_cp_comparison.py:
estimate_noise_valus_for_policies) already writes, per checkpoint pair, the
null mean/std of each of the three evolutionary metrics across M split-half
resamples, and the p90/p95/p99 percentiles of the null MAX-statistic
(zmax = max_j Z_j across the 3 metrics). This script is the missing
downstream step: it z-scores the OBSERVED (real, non-null) metric values
against that same null mean/std, takes the max across the 3 metrics, and
compares it to zmax_p95 (i.e. a family-wise alpha=0.05 test), which is the
statistic the paper's methodology (Section on the noise baseline) actually
describes -- NOT a union/OR of the three per-metric individual thresholds.

Requires zmax_p95 / null_mean_* / null_std_* to be populated in the metrics
json (only true for runs produced by the M=20 pipeline after these fields
were added; legacy M=2 files predate them and are skipped).
"""
import json
import glob
from statistics import mean

DECISION_METRICS = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None


def family_wise_active(d):
    """Returns (n_active, n_total) or None if the file lacks the null fields."""
    zmax_p95 = d.get("zmax_p95")
    if not isinstance(zmax_p95, list) or not zmax_p95 or zmax_p95[0] is None:
        return None

    raws = {k: d.get(k + "_raw") for k in DECISION_METRICS}
    means = {k: d.get("null_mean_" + k) for k in DECISION_METRICS}
    stds = {k: d.get("null_std_" + k) for k in DECISION_METRICS}
    n = len(zmax_p95)
    if any(not isinstance(raws[k], list) or len(raws[k]) < n for k in DECISION_METRICS):
        return None

    active = 0
    for i in range(n):
        zs = []
        for k in DECISION_METRICS:
            mu, sd = means[k][i], stds[k][i]
            if sd is None or sd == 0:
                continue
            zs.append((raws[k][i] - mu) / sd)
        if not zs:
            continue
        if max(zs) > zmax_p95[i]:
            active += 1
    return active, n


if __name__ == "__main__":
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "results/**/*_metrics.json"
    per_env = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = load(f)
        if not d:
            continue
        res = family_wise_active(d)
        if res is None:
            continue
        env = f.replace("\\", "/").split("/")[1]
        per_env.setdefault(env, []).append((f, res))

    for env, items in per_env.items():
        print(f"\n=== {env} ===")
        tot_a, tot_n = 0, 0
        for f, (a, n) in items:
            print(f"  {a}/{n}  {f}")
            tot_a += a
            tot_n += n
        print(f"  POOLED: {tot_a}/{tot_n} = {tot_a/tot_n:.3f}")

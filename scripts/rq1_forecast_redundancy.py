#!/usr/bin/env python3
"""Reconstruct, on metric_results, the two RQ1 numbers whose original aggregators were lost:
  (A) the run-level forecast: among runs flat over the first half of training, does first-half
      fingerprint activity separate later recoverers from the permanently stalled? (AUC)
  (B) metric non-redundancy: over all induced-mode pairs, how often is each evolutionary metric
      the SOLE one above its own noise floor?

USAGE: python scripts/rq1_forecast_redundancy.py
"""
import glob, hashlib, json, math, os
import numpy as np
from scipy.stats import rankdata

ROOT = r"C:\Users\ondra\Documents\metric_results"
EVO = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]
ENV_DIR = {"FrozenLake": "frozen_lake8x8", "MountainCar": "mountain_car", "Taxi": "taxi", "NetSecGame": "netsecgame"}
ENV_N = {"FrozenLake": 500, "MountainCar": 500, "Taxi": 1000, "NetSecGame": 200}
TABULAR = {"q_learning", "sarsa"}
Z_CRIT = 1.9599639845400545
EXCL = ("30K", "bins15", "bins30", "_bak")


def _arr(d, k):
    return np.array([np.nan if v is None else float(v) for v in d.get(k, [])], float)


def fp_active_series(d):
    """Per-pair family-wise fingerprint decision (max z > zmax_p95). NaN where undefined."""
    zp = _arr(d, "zmax_p95"); n = len(zp)
    if n == 0:
        return np.array([])
    zs = []
    for k in EVO:
        raw, mu, sd = _arr(d, k + "_raw"), _arr(d, "null_mean_" + k), _arr(d, "null_std_" + k)
        if len(raw) < n:
            return np.array([])
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (raw - mu) / sd
        z[sd == 0] = np.nan
        zs.append(z[:n])
    return (np.nanmax(np.vstack(zs), axis=0) > zp).astype(float)


def ret_changed_series(d, n_eval):
    mr, sr = _arr(d, "mean_return"), _arr(d, "std_return")
    out = []
    for i in range(len(mr) - 1):
        se2 = sr[i] ** 2 / n_eval + sr[i + 1] ** 2 / n_eval
        z = (mr[i + 1] - mr[i]) / math.sqrt(se2) if se2 > 0 else (0.0 if mr[i + 1] == mr[i] else math.copysign(9e9, mr[i + 1] - mr[i]))
        out.append(z)
    return np.array(out), mr


def auc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def unique_runs(env_glob):
    """All non-variant metric files under a glob, deduped by (mean_return, perplexity)."""
    seen = set()
    for f in sorted(glob.glob(env_glob, recursive=True)):
        if any(t in f for t in EXCL):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sig = hashlib.md5((json.dumps(d.get("mean_return")) + json.dumps(d.get("state_visitation_perplexity"))).encode()).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        yield f, d


# ---------------- (A) run-level forecast ----------------
def _zpair(mr, sr, i, j, N):
    """z of the two-sample test between checkpoints i and j (signed, up = positive)."""
    se2 = sr[i] ** 2 / N + sr[j] ** 2 / N
    return (mr[j] - mr[i]) / math.sqrt(se2) if se2 > 0 else (0.0 if mr[j] == mr[i] else math.copysign(9e9, mr[j] - mr[i]))


def forecast():
    print("=== (A) run-level forecast: first-half activity separates recoverers from stalled ===")
    rows = []  # (is_tabular, recovered, first_half_activity)
    for envname, ed in ENV_DIR.items():
        for f, d in unique_runs(os.path.join(ROOT, ed, "*", "standard", "**", "*_metrics.json")):
            parts = os.path.relpath(f, ROOT).split(os.sep)
            algo = parts[1]
            fp = fp_active_series(d)
            if fp.size < 4:
                continue
            _, mr = ret_changed_series(d, ENV_N[envname]); sr = _arr(d, "std_return")
            if len(mr) < len(fp) + 1:
                continue
            mid = len(fp) // 2
            if mid < 2:
                continue
            # flat first half: the return has not significantly RISEN from the start by mid-training
            if _zpair(mr, sr, 0, mid, ENV_N[envname]) > Z_CRIT:
                continue
            recovered = _zpair(mr, sr, mid, len(mr) - 1, ENV_N[envname]) > Z_CRIT   # rises in the second half
            rows.append((algo in TABULAR, recovered, float(fp[:mid].mean())))
    rows = np.array([(t, r, a) for (t, r, a) in rows], float)
    n = len(rows); rec = int(rows[:, 1].sum())
    print(f"  flat-first-half runs n={n}  (recoverers {rec}, stalled {n-rec})")
    for lab, mask in [("tabular", rows[:, 0] == 1), ("deep", rows[:, 0] == 0), ("all", np.ones(n, bool))]:
        sub = rows[mask]; pos = sub[sub[:, 1] == 1, 2]; neg = sub[sub[:, 1] == 0, 2]
        print(f"  {lab:8s} n={len(sub):3d}  recoverers={len(pos):2d} stalled={len(neg):2d}  "
              f"median act rec={np.median(pos) if len(pos) else float('nan'):.2f} vs stall={np.median(neg) if len(neg) else float('nan'):.2f}  "
              f"AUC={auc(neg, pos):.2f}")


# ---------------- (B) metric non-redundancy ----------------
def non_redundancy():
    print("\n=== (B) metric non-redundancy: sole-active counts over all induced-mode pairs ===")
    ge1 = alltot = 0
    sole = {k: 0 for k in EVO}; sole_es = {k: 0 for k in EVO}
    for envname, ed in ENV_DIR.items():
        for f, d in unique_runs(os.path.join(ROOT, ed, "**", "*_metrics.json")):
            raws = {k: _arr(d, k + "_raw") for k in EVO}
            ths = {k: _arr(d, k + "_noise_threshold") for k in EVO}
            n = min(len(raws[k]) for k in EVO)
            if n == 0 or any(len(ths[k]) < n for k in EVO):
                continue
            for i in range(n):
                if any(np.isnan(raws[k][i]) or np.isnan(ths[k][i]) for k in EVO):
                    continue                                                 # need all three defined
                alltot += 1
                act = {k: raws[k][i] > ths[k][i] for k in EVO}
                if any(act.values()):
                    ge1 += 1
                if sum(act.values()) == 1:
                    only = [k for k in EVO if act[k]][0]
                    sole[only] += 1
                    if raws[only][i] > 0.01:
                        sole_es[only] += 1
    print(f"  pairs with all three defined: {alltot}   at least one active: {ge1}")
    lab = {"topological_shift": "Topological", "strategic_shift": "Strategic", "3-gram_wasserstein": "3-gram (Seq)"}
    print("  sole-active:        " + "  ".join(f"{lab[k]}={sole[k]}" for k in EVO))
    print("  sole-active raw>.01:" + "  ".join(f"{lab[k]}={sole_es[k]}" for k in EVO))


if __name__ == "__main__":
    forecast()
    non_redundancy()

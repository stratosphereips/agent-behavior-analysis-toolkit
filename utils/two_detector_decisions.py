#!/usr/bin/env python3
"""
Two-detector per-checkpoint-pair decision, per Section 4.6.1 (RQ1 sensitivity).

For each run (one row of the input CSV) this reads the run's metrics JSON and,
for every consecutive checkpoint pair, decides the joint state of the two
detectors:

  - Return detector : the mean return CHANGED between the two checkpoints, by a
                      two-sample, two-sided z-test at p < 0.05 (|z| > 1.96).
  - Fingerprint (BF): the fingerprint is ACTIVE, by the family-wise max-statistic
                      of Section 4.6.1 / the noise-baseline section: standardise
                      each of the three evolutionary metrics against its own
                      permutation null (z_j = (m_j - mu_j)/sigma_j) and compare
                      max_j z_j against the 95th percentile of the null maximum
                      (zmax_p95). This is NOT a per-metric OR (that inflates the
                      family-wise error to ~14%); the OR is only available behind
                      --union-fallback for legacy files that lack the null fields.

Per-pair decision is one of:
    both active | BF Only | Reward only | None

INPUT CSV  (with or without a header row):
    env, model, seed, path_to_metrics_json [, n]
    e.g.  Taxi,dqn,4242,results/taxi/dqn/standard/...metrics.json

OUTPUT CSV (to stdout):
    env, model, seed, <decision cp1>, <decision cp2>, ...
    e.g.  Taxi,dqn,4242,None,BF Only,BF Only,both active,...

USAGE:
    python scripts/two_detector_decisions.py runs.csv --n 500 > decisions.csv
    python scripts/two_detector_decisions.py runs.csv --default-n 500 > decisions.csv
    python scripts/two_detector_decisions.py runs.csv --n 500 --union-fallback > decisions.csv

The return z-test needs the number of evaluation episodes N (std_return is the
per-episode std, so SE = std/sqrt(N)). N is resolved per row in priority order:
  --n (runtime, applies to all rows) -> 5th CSV column -> 'evaluate_for=<N>' in
  the path -> --default-n (fallback).
If none resolve, the row is skipped with an error on stderr.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

Z_CRIT = 1.9599639845400545  # two-sided p<0.05
EVO_METRICS = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]

BOTH, BF, REW, NONE = "both active", "BF Only", "Reward only", "None"


def warn(msg):
    print(f"[two_detector] {msg}", file=sys.stderr)


def resolve_n(row_n, path, override_n, default_n):
    # Runtime --n overrides everything; then per-row column, then path, then --default-n.
    if override_n:
        return override_n
    if row_n:
        try:
            return int(row_n)
        except ValueError:
            pass
    m = re.search(r"evaluate_for=(\d+)", path)
    if m:
        return int(m.group(1))
    return default_n


def return_changed(m1, s1, m2, s2, n):
    """Two-sample, two-sided z-test on the means; True if the return changed."""
    se2 = (s1 * s1) / n + (s2 * s2) / n
    if se2 <= 0.0:
        return m1 != m2  # degenerate: zero variance both sides
    z = (m2 - m1) / math.sqrt(se2)
    return abs(z) > Z_CRIT


def fp_active_maxstat(d, i):
    """Family-wise max-statistic decision for pair i. None if null fields absent."""
    zmax_p95 = d.get("zmax_p95")
    if not (isinstance(zmax_p95, list) and i < len(zmax_p95) and zmax_p95[i] is not None):
        return None
    zs = []
    for k in EVO_METRICS:
        raw = d.get(k + "_raw")
        mu = d.get("null_mean_" + k)
        sd = d.get("null_std_" + k)
        if not all(isinstance(x, list) and i < len(x) for x in (raw, mu, sd)):
            return None
        if mu[i] is None or sd[i] is None or sd[i] == 0:
            continue
        z = (raw[i] - mu[i]) / sd[i]
        if math.isfinite(z):
            zs.append(z)
    if not zs:
        return False
    return max(zs) > zmax_p95[i]


def fp_active_union(d, i):
    """Per-metric OR against individual noise floors (BIASED; --union-fallback only)."""
    for k in EVO_METRICS:
        raw = d.get(k + "_raw")
        thr = d.get(k + "_noise_threshold")
        if isinstance(raw, list) and isinstance(thr, list) and i < len(raw) and i < len(thr):
            if raw[i] > thr[i]:
                return True
    return False


def process_run(path, n, union_fallback):
    """Returns (decisions_list, note) or (None, error_str)."""
    if not os.path.isfile(path):
        return None, f"file not found: {path}"
    try:
        d = json.load(open(path))
    except Exception as e:
        return None, f"could not read json ({e}): {path}"

    mean_r = d.get("mean_return")
    std_r = d.get("std_return")
    n_pairs = len(d.get("topological_shift_raw", []))
    if n_pairs == 0 or not isinstance(mean_r, list) or len(mean_r) < n_pairs + 1:
        return None, f"missing/short return or metric arrays: {path}"

    decisions = []
    used_fallback = False
    for i in range(n_pairs):
        rc = return_changed(mean_r[i], std_r[i], mean_r[i + 1], std_r[i + 1], n)
        fa = fp_active_maxstat(d, i)
        if fa is None:
            if union_fallback:
                fa = fp_active_union(d, i)
                used_fallback = True
            else:
                return None, ("missing family-wise null fields (zmax_p95/null_mean_*/"
                              f"null_std_*); regenerate via the M-split pipeline, or pass "
                              f"--union-fallback for the (biased) per-metric OR: {path}")
        if rc and fa:
            decisions.append(BOTH)
        elif fa:
            decisions.append(BF)
        elif rc:
            decisions.append(REW)
        else:
            decisions.append(NONE)
    return decisions, ("union-fallback" if used_fallback else "")


def looks_like_header(fields):
    if len(fields) < 4:
        return False
    joined = ",".join(fields).lower()
    return ("path" in joined or "env" in joined and "model" in joined) and not os.path.sep in fields[3] and not fields[3].endswith(".json")


def main():
    ap = argparse.ArgumentParser(description="Two-detector per-CP decisions (Sec 4.6.1).")
    ap.add_argument("input_csv", help="CSV: env,model,seed,path[,n]")
    ap.add_argument("-n", "--n", type=int, default=None,
                    help="Eval-episode count N applied to ALL rows (highest precedence, "
                         "overrides a per-row column or path-parsed value).")
    ap.add_argument("--default-n", type=int, default=None,
                    help="Fallback N used only when N is not given by --n, a 5th CSV column, "
                         "or an 'evaluate_for=' token in the path.")
    ap.add_argument("--union-fallback", action="store_true",
                    help="For files lacking the family-wise null fields, fall back to the "
                         "(biased) per-metric OR rule. Non-canonical; use only for legacy files.")
    ap.add_argument("out_csv", nargs="cp_raster.csv", default="", help="Output CSV (default stdout)")
    args = ap.parse_args()

    out = csv.writer(sys.stdout)
    with open(args.input_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        warn("empty input CSV")
        return
    start = 1 if looks_like_header([c.strip() for c in rows[0]]) else 0

    for r in rows[start:]:
        r = [c.strip() for c in r]
        if not r or len(r) < 4:
            continue
        env, model, seed, path = r[0], r[1], r[2], r[3]
        row_n = r[4] if len(r) > 4 else None
        n = resolve_n(row_n, path, args.n, args.default_n)
        if not n:
            warn(f"SKIP {env},{model},{seed}: no N (add a 5th CSV column, put "
                 f"evaluate_for= in the path, or pass --default-n): {path}")
            continue
        decisions, note = process_run(path, n, args.union_fallback)
        if decisions is None:
            warn(f"SKIP {env},{model},{seed}: {note}")
            continue
        if note:
            warn(f"NOTE {env},{model},{seed}: {note} (N={n})")
        out.writerow([env, model, seed] + decisions)


if __name__ == "__main__":
    main()

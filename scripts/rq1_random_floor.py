#!/usr/bin/env python3
"""
RQ1 empirical false-positive floor: the fraction of checkpoint pairs the
family-wise fingerprint decision flags ACTIVE on a non-learning policy, pooled
per environment. Same max-statistic rule as two_detector_decisions.py (every
"active" pair on a stationary policy is a false positive).

You provide the environments and their data sources as two parallel lists:

    python scripts/rq1_random_floor.py \
        --envs Taxi MountainCar FrozenLake \
        --src_data path/to/taxi/random path/to/mc/random path/to/fl/random

Each --src_data entry (one per --envs entry) may be:
  * a directory  -> recursively globbed for '*_metrics.json'
  * a glob        -> e.g. 'results/mc/random/**/seed_*_20_metrics.json'
  * a single file -> a metrics.json
  * a comma-separated list of any of the above (e.g. 'a.json,b.json')

Files that lack the family-wise null fields (zmax_p95/null_mean_*/null_std_*)
are reported as SKIP (they predate the M-split pipeline and need regenerating).
"""
import argparse
import glob
import json
import math
import os

EVO = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]


def fp_active(d, i):
    """Family-wise max-statistic decision for pair i; None if null fields absent."""
    p95 = d.get("zmax_p95")
    if not (isinstance(p95, list) and i < len(p95) and p95[i] is not None):
        return None
    zs = []
    for k in EVO:
        raw, mu, sd = d.get(k + "_raw"), d.get("null_mean_" + k), d.get("null_std_" + k)
        if not all(isinstance(x, list) and i < len(x) for x in (raw, mu, sd)):
            return None
        if mu[i] is None or sd[i] is None or sd[i] == 0:
            continue
        z = (raw[i] - mu[i]) / sd[i]
        if math.isfinite(z):
            zs.append(z)
    if not zs:
        return False
    return max(zs) > p95[i]


def floor_for_file(path):
    """(active_pairs, total_pairs) or None if the file lacks null fields."""
    try:
        d = json.load(open(path))
    except Exception as e:
        return ("error", str(e))
    n = len(d.get("topological_shift_raw", []))
    active = 0
    for i in range(n):
        a = fp_active(d, i)
        if a is None:
            return None
        active += int(a)
    return active, n


def expand_source(src):
    """Turn one --src_data token into a list of metrics.json paths."""
    files = []
    for part in src.split(","):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            files += sorted(glob.glob(os.path.join(part, "**", "*_metrics.json"), recursive=True))
        elif any(ch in part for ch in "*?["):
            files += sorted(glob.glob(part, recursive=True))
        elif os.path.isfile(part):
            files.append(part)
        else:
            print(f"[floor] WARN: source not found: {part}")
    return files


def main():
    ap = argparse.ArgumentParser(description="RQ1 family-wise false-positive floor per environment.")
    ap.add_argument("--envs", nargs="+", required=True, help="Environment names.")
    ap.add_argument("--src_data", nargs="+", required=True,
                    help="Data source per env (dir / glob / file / comma-list), parallel to --envs.")
    args = ap.parse_args()

    if len(args.envs) != len(args.src_data):
        ap.error(f"--envs ({len(args.envs)}) and --src_data ({len(args.src_data)}) must have equal length.")

    for env, src in zip(args.envs, args.src_data):
        print(f"\n=== {env} ===")
        files = expand_source(src)
        if not files:
            print("  (no metrics.json found)")
            continue
        tot_a = tot_n = 0
        for p in files:
            r = floor_for_file(p)
            disp = p.replace(os.sep, "/")
            if r is None:
                print(f"  SKIP (no null fields): {disp}")
            elif r[0] == "error":
                print(f"  SKIP (read error: {r[1]}): {disp}")
            else:
                a, n = r
                tot_a += a
                tot_n += n
                print(f"  {a}/{n}  {disp}")
        if tot_n:
            print(f"  POOLED FLOOR: {tot_a}/{tot_n} = {tot_a / tot_n:.3f}")
        else:
            print("  POOLED FLOOR: n/a (no scorable files)")


if __name__ == "__main__":
    main()

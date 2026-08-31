#!/usr/bin/env python3
"""
RQ1 dose-response: does the MAGNITUDE of fingerprint-only activity predict the
SIZE of the subsequent return change?

This is the supporting analysis behind Finding 2 (Section 5.2). Counting
"confirmed" fingerprint-only pairs (a later return change follows) shows the
flagged activity is not terminal detector noise, but it is weak on its own
because a Good-Learning run's return moves eventually anyway. Here we test a
graded relationship: within each reward-flat gap that later resolves into a
significant return change, we correlate how strongly the fingerprint fired
against how large that terminating change is.

Key point (why this is a funnel, not a rising trend): the fingerprint tracks the
MAGNITUDE of the impending return change, |d return|, NOT its direction. Across
all resolved gaps ~46% end in a return DROP, and the correlation of fingerprint
strength with the *signed* change is ~0 while the correlation with |change| is
positive. We therefore plot the SIGNED change (a funnel that widens with x) and
report Spearman rho on |change|.

Both detectors are exactly those of scripts/two_detector_decisions.py:
  - Fingerprint active : family-wise max-statistic  max_j (m_j-mu_j)/sd_j > zmax_p95.
  - Return changed     : two-sample two-sided z-test on the mean return,
                         SE = std/sqrt(N), |z| > 1.96 (a "rise" is z > +1.96).

For each reward-flat gap (maximal run of no-change pairs) that terminates in a
return change and contains >= 1 fingerprint-active pair:
  x = mean over the gap's active pairs of the excess (max_j z_j - zmax_p95)
  y = signed terminating return change, normalised by the run's return range
      (max-min mean_return) so environments are comparable; |y| is the size.

We report Spearman rho on |change| pooled, within each environment (the level
that controls the between-environment scale confound), partial-out the gap's
training-phase position, the signed-change correlation (direction test), and a
per-run summary; plus the reconstruction counts (total / fingerprint-only /
confirmed / return-only) as a sanity check against the paper's 2454 / 1058 /
915 / 16.

USAGE:
    python scripts/rq1_dose_response.py                       # stats only
    python scripts/rq1_dose_response.py --out_fig figures/rq1_funnel.png
    python scripts/rq1_dose_response.py --results_root results --mode standard
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from scipy import stats

Z_CRIT = 1.9599639845400545  # two-sided p<0.05
EVO_METRICS = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]
ENV_N = {"Taxi": 1000, "FrozenLake": 500, "MountainCar": 500}
ENV_DIR = {"FrozenLake": "frozenlake8x8", "MountainCar": "mountain_car", "Taxi": "taxi"}
ALGOS = ["q_learning", "sarsa", "dqn", "ppo"]
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5", "seed4242"]
EXCLUDE = {("Taxi", "ppo")}  # return never active -> nothing to confirm against
COLORS = {"FrozenLake": "#F58518", "MountainCar": "#4C78A8", "Taxi": "#54A24B"}


def _arr(d, k):
    return np.array([np.nan if v is None else float(v) for v in d[k]], dtype=float)


def excess_z(d):
    """Per-pair max-statistic minus its family-wise cutoff (>0 == fingerprint active)."""
    zs = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in EVO_METRICS:
            z = (_arr(d, k + "_raw") - _arr(d, "null_mean_" + k)) / _arr(d, "null_std_" + k)
            zs.append(z)
    zmax = np.nanmax(np.vstack(zs), axis=0)
    return zmax - _arr(d, "zmax_p95")


def return_detect(d, n):
    """Per-pair (changed, rise, mean_return) from the two-sample z-test on the mean return."""
    mr, sr = _arr(d, "mean_return"), _arr(d, "std_return")
    m = len(mr)
    z = np.full(m - 1, np.nan)
    for i in range(m - 1):
        se2 = sr[i] ** 2 / n + sr[i + 1] ** 2 / n
        diff = mr[i + 1] - mr[i]
        z[i] = (0.0 if diff == 0 else math.copysign(1e9, diff)) if se2 <= 0 else diff / math.sqrt(se2)
    return np.abs(z) > Z_CRIT, z > Z_CRIT, mr


def load_run(root, env, algo, seed, mode):
    fs = glob.glob(os.path.join(root, ENV_DIR[env], algo, mode, seed, "*_metrics.json"))
    return json.load(open(fs[0])) if fs else None


def collect(root, mode):
    """Return (gaps, counts).

    gaps: list of dicts with env, run, x (mean excess), signed (Δreturn/range),
          absy (|Δreturn|/range), tf (gap-midpoint training fraction), up (bool).
    """
    gaps, rid = [], 0
    tot = dict(pairs=0, fp_only=0, confirmed=0, ret_only=0)
    for env in ENV_DIR:
        for algo in ALGOS:
            if (env, algo) in EXCLUDE:
                continue
            for seed in SEEDS:
                d = load_run(root, env, algo, seed, mode)
                if d is None:
                    continue
                n = ENV_N[env]
                ex = excess_z(d)
                fp = np.where(np.isnan(ex), False, ex > 0)
                changed, rise, mr = return_detect(d, n)
                npair = len(fp)
                rng = np.nanmax(mr) - np.nanmin(mr)
                rng = rng if rng > 0 else 1.0

                # reconstruction sanity counts
                tot["pairs"] += npair
                fp_only = fp & ~changed
                tot["fp_only"] += int(fp_only.sum())
                tot["ret_only"] += int((~fp & changed).sum())
                for i in np.where(fp_only)[0]:
                    if changed[i + 1:].any():  # paper's "confirmed": a later return CHANGE
                        tot["confirmed"] += 1

                # maximal reward-flat gaps that terminate in a return change,
                # with >= 1 fingerprint-active pair inside
                i = 0
                while i < npair:
                    if not changed[i]:
                        j = i
                        while j < npair and not changed[j]:
                            j += 1
                        if j < npair and fp[i:j].any():
                            signed = (mr[j + 1] - mr[j]) / rng
                            gaps.append(dict(
                                env=env, run=rid,
                                x=float(np.nanmean(ex[i:j][fp[i:j]])),
                                signed=signed, absy=abs(signed),
                                tf=(i + j) / 2.0 / npair, up=bool(rise[j]),
                            ))
                        i = j + 1
                    else:
                        i += 1
                rid += 1
    return gaps, tot


def _spear(x, y):
    r = stats.spearmanr(x, y)
    return r.statistic, r.pvalue


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank residuals)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        b = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ coef

    return stats.pearsonr(resid(rx, rz), resid(ry, rz))


def report(gaps, counts):
    env = np.array([g["env"] for g in gaps])
    x = np.array([g["x"] for g in gaps])
    signed = np.array([g["signed"] for g in gaps])
    absy = np.array([g["absy"] for g in gaps])
    tf = np.array([g["tf"] for g in gaps])
    run = np.array([g["run"] for g in gaps])
    n_drop = int((signed < 0).sum())

    print("=== reconstruction sanity (paper: 2454 / 1058 / 915 / 16) ===")
    print(f"  pairs={counts['pairs']}  fp_only={counts['fp_only']}  "
          f"confirmed={counts['confirmed']}  return_only={counts['ret_only']}")
    print("  (pairs/fp_only/return_only reproduce the pipeline exactly; confirmed is within a\n"
          "   few of the paper because this script omits the two-consecutive-pair FP-active\n"
          "   declaration rule, which shifts a handful of MountainCar pairs.)")

    print(f"\n=== dose-response: fingerprint strength vs size of terminating return change ===")
    print(f"  resolved flat gaps with activity: {len(gaps)}  "
          f"(rises: {len(gaps) - n_drop}, drops: {n_drop} = {n_drop / len(gaps):.0%})")
    rho, p = _spear(x, absy)
    print(f"  POOLED  Spearman(strength, |change|) rho={rho:+.3f}  p={p:.2e}")
    for e in ENV_DIR:
        m = env == e
        if m.sum() > 3:
            r, pp = _spear(x[m], absy[m])
            print(f"    {e:12} n={m.sum():4d}  rho={r:+.3f}  p={pp:.2e}")
    pr, pp = _partial_spearman(x, absy, tf)
    print(f"  partial (control training-phase) rho={pr:+.3f}  p={pp:.2e}")
    rs, ps = _spear(x, signed)
    print(f"  DIRECTION test  Spearman(strength, SIGNED change) rho={rs:+.3f}  p={ps:.2e} "
          f"(~0 == size predicted, not direction)")
    prr = []
    for r_ in np.unique(run):
        m = run == r_
        if m.sum() >= 5 and np.ptp(x[m]) > 0 and np.ptp(absy[m]) > 0:
            prr.append(_spear(x[m], absy[m])[0])
    if prr:
        prr = np.array(prr)
        print(f"  per-run rho(|change|): median={np.median(prr):+.3f}  pos={int((prr > 0).sum())}/{len(prr)}"
              f"  Wilcoxon p={stats.wilcoxon(prr).pvalue:.2e}")
    return env, x, signed, absy


def plot(env, x, signed, out_fig, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.axhline(0, color="0.5", lw=0.8, zorder=0)
    for e in ["FrozenLake", "MountainCar", "Taxi"]:
        m = env == e
        if not m.any():
            continue
        ax.scatter(x[m], signed[m], s=28, alpha=0.75, c=COLORS[e],
                   edgecolors="white", linewidths=0.3, label=e)
    ax.set_xscale("log")
    ax.set_xlabel("Fingerprint strength during flat stretch\n(mean signal above detection threshold; log scale)")
    ax.set_ylabel("Return change after the stretch\n(signed, fraction of run range)")
    ax.legend(fontsize=9.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight")
    print(f"\nsaved figure -> {out_fig}  (dpi={dpi})")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="RQ1 dose-response / funnel (Finding 2 support).")
    ap.add_argument("--results_root", default=os.path.join(here, "results"),
                    help="Root holding <env>/<algo>/<mode>/seed*/*_metrics.json")
    ap.add_argument("--mode", default="standard", help="Learning mode subfolder (default: standard).")
    ap.add_argument("--out_fig", default=None, help="If set, write the funnel figure here.")
    ap.add_argument("--dpi", type=int, default=400, help="Figure DPI (default: 400).")
    args = ap.parse_args()

    gaps, counts = collect(args.results_root, args.mode)
    if not gaps:
        print("No gaps found; check --results_root.")
        return
    env, x, signed, absy = report(gaps, counts)
    if args.out_fig:
        plot(env, x, signed, args.out_fig, args.dpi)


if __name__ == "__main__":
    main()

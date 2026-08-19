#!/usr/bin/env python3
"""RQ1 lead-time strip plot.

For each run, lead = (first_reward_cp - first_fp_cp) / total_cps.
  Positive  →  fingerprint fires first  (fingerprint leads).
  Zero      →  both fire at the same checkpoint.
  Negative  →  reward fires first  (reward leads).

One subplot per environment; algorithms on the y-axis; each seed is one dot.

Usage:
    python plot_lead_strip.py agreement.csv -o rq1_leadstrip.png
"""

import argparse
import csv
import random
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP_STATES = {"bf only", "both active"}
RW_STATES = {"reward only", "both active"}
KNOWN     = {"bf only", "both active", "reward only", "none"}

ENV_PALETTE = ["#3B7DD8", "#E8720C", "#5AA02C", "#9B59B6", "#C0392B", "#1ABC9C"]
JITTER_Y    = 0.12


def model_type(name):
    return re.sub(r"[_\-\s]?seed[_\-\s]?\d+$", "", name.strip(), flags=re.I) or name.strip()


def first_sustained(flags, min_run):
    n = len(flags)
    for i in range(n - min_run + 1):
        if all(flags[i:i + min_run]):
            return i
    return None


def read_runs(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        sys.exit("empty CSV")
    header = [h.strip().lower() for h in rows[0]]
    env_i  = header.index("env")        if "env"        in header else 0
    mdl_i  = (header.index("model_name") if "model_name" in header
               else header.index("model") if "model"     in header
               else 1)
    seed_i = header.index("seed") if "seed" in header else None
    meta   = {env_i, mdl_i} | ({seed_i} if seed_i is not None else set())
    cp_cols = [i for i in range(len(rows[0])) if i not in meta]

    runs = []
    for r in rows[1:]:
        r = r + [""] * (len(rows[0]) - len(r))
        env   = r[env_i].strip()
        mdl   = r[mdl_i].strip()
        cells = [r[i].strip().lower() for i in cp_cols]
        for c in cells:
            if c and c not in KNOWN:
                sys.stderr.write(f"warning: unrecognised cell '{c}'\n")
        runs.append((env, mdl, cells))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Lead-time strip plot per environment.")
    ap.add_argument("csv_path")
    ap.add_argument("-o", "--out", default="rq1_leadstrip.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--reward-min-run", type=int, default=1)
    ap.add_argument("--fp-min-run",     type=int, default=1)
    args = ap.parse_args()

    runs = read_runs(args.csv_path)

    # compute lead values grouped by env → algo
    env_data = OrderedDict()   # env → OrderedDict(algo → [lead, ...])
    skipped  = []

    for env, mdl, cells in runs:
        valid = [i for i, c in enumerate(cells) if c != ""]
        if len(valid) < 2:
            skipped.append(mdl); continue
        total_cps = len(valid)   # actual checkpoints for this run; padding columns excluded
        fp_flags  = [c in FP_STATES for c in cells]
        rw_flags  = [c in RW_STATES for c in cells]
        fp_i = first_sustained(fp_flags, args.fp_min_run)
        rw_i = first_sustained(rw_flags, args.reward_min_run)
        if fp_i is None or rw_i is None:
            skipped.append(mdl); continue
        norm = total_cps if total_cps > 1 else 1   # fraction of the run's own checkpoints (matches docstring)
        lead = rw_i / norm - fp_i / norm
        mt   = model_type(mdl)
        if mt.strip().lower() == "random":
            continue
        env_data.setdefault(env, OrderedDict()).setdefault(mt, []).append(lead)

    if skipped:
        sys.stderr.write(f"skipped {len(skipped)} run(s): {', '.join(skipped)}\n")
    if not env_data:
        sys.exit("no plottable runs")

    envs       = list(env_data.keys())
    n_envs     = len(envs)
    env_colors = {e: ENV_PALETTE[i % len(ENV_PALETTE)] for i, e in enumerate(envs)}

    # consistent algo ordering across all envs (first-seen)
    all_algos = list(OrderedDict.fromkeys(
        algo for ed in env_data.values() for algo in ed
    ))
    algo_y  = {a: i for i, a in enumerate(all_algos)}
    n_algos = len(all_algos)

    x_all = [v for ed in env_data.values() for vals in ed.values() for v in vals]
    x_abs = max(abs(v) for v in x_all) * 1.25 if x_all else 0.5

    fig, axes = plt.subplots(
        1, n_envs,
        figsize=(4.5 * n_envs, max(3.0, 1.0 + n_algos * 0.7)),
        sharey=True,
    )
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        color      = env_colors[env]
        algo_leads = env_data[env]

        ax.axvspan(0, x_abs, color="#5AA02C", alpha=0.06, zorder=0)
        ax.axvline(0, color="#888888", lw=1.0, ls="--", zorder=1)

        for algo, leads in algo_leads.items():
            yi = algo_y[algo]
            for lead in leads:
                jy = yi + random.uniform(-JITTER_Y, JITTER_Y)
                ax.scatter(lead, jy, c=color, s=40, alpha=0.85,
                           edgecolors="none", zorder=3)

        ax.set_xlim(-x_abs, x_abs)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.text( x_abs * 0.55, n_algos - 0.7, "fingerprint\nleads",
                color="#4A7A25", fontsize=8, style="italic", ha="center")
        ax.text(-x_abs * 0.55, n_algos - 0.7, "reward\nleads",
                color="#999999", fontsize=8, style="italic", ha="center")

    axes[0].set_yticks(range(n_algos))
    axes[0].set_yticklabels(all_algos, fontsize=9)
    axes[0].set_ylim(-0.5, n_algos - 0.5)

    axes[n_envs // 2].set_xlabel(
        "lead time  (reward fraction − fingerprint fraction)", fontsize=10)

    if args.title:
        fig.suptitle(args.title, fontsize=13)

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    n_runs = sum(len(v) for ed in env_data.values() for v in ed.values())
    print(f"wrote {args.out}  ({n_runs} runs, {n_envs} environments, {n_algos} algorithms)")


if __name__ == "__main__":
    main()

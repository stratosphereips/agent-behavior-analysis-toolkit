"""
Plot N sensitivity results produced by run_n_sensitivity.py.

Each input JSON corresponds to one environment/algorithm run. The script
aggregates a chosen stability field across all checkpoint pairs within each
file, then plots mean +/- std per metric as a function of N (the number of
evaluation trajectories).

The primary field is `std_over_range` (bootstrap std normalized by the metric's
signal range across the checkpoint series): lower = more stable, and it is
directly comparable across metrics and robust to means that are near or cross
zero. The claim the figure supports is that the behavioral metrics drop below a
given stability tolerance at a *smaller* N than return does.

Usage:
    python experiments/plot_n_sensitivity.py \
        --inputs results/n_sensitivity_frozenlake_bootstrap.json \
                 results/n_sensitivity_mountaincar_bootstrap.json \
                 results/n_sensitivity_taxi_bootstrap.json \
        --labels FrozenLake MountainCar Taxi \
        --mode curve --output figures/n_sensitivity_curve.png \
        --field std_over_range \
        --tolerance 0.1

    python experiments/plot_n_sensitivity.py \
        --inputs results/n_sensitivity_frozenlake_bootstrap.json \
                 results/n_sensitivity_mountaincar_bootstrap.json \
                 results/n_sensitivity_taxi_bootstrap.json \
        --labels FrozenLake MountainCar Taxi \
        --mode operating --output figures/n_sensitivity_operating.png \
        --field std_over_range \
        --tolerance 0.1
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

METRICS = [
    ("seff",                    r"$\mathrm{PP}(\hat{P}_k)$"),
    ("topological_shift",       r"$\Delta$Topo"),
    ("strategic_shift",         r"$\Delta$Strat"),
    ("3-gram_wasserstein",      r"$\Delta$Seq"),
    ("mean_return",             "Return"),
]

METRIC_COLORS = {
    "seff":                "#9C27B0",   # purple (matches perplexity panels)
    "topological_shift":   "#2196F3",   # blue
    "strategic_shift":     "#FF9800",   # orange
    "3-gram_wasserstein":  "#4CAF50",   # green
    "mean_return":         "#E91E63",   # pink/red — return baseline
}

METRIC_LINESTYLE = {
    "mean_return": "--",                 # dashed to visually separate return
}


def load_and_aggregate(path: str, field: str) -> dict[str, dict[int, list]]:
    """
    Load a sensitivity JSON and aggregate the chosen field across checkpoint pairs.

    Handles both the wrapped format ({"pairs": {...}, "signal_range": {...}}) and
    a bare pair-keyed dict, for forward/backward compatibility.

    Returns:
        {metric: {N: [field values across pairs]}}
    """
    with open(path) as f:
        data = json.load(f)

    pairs = data["pairs"] if isinstance(data, dict) and "pairs" in data else data

    aggregated: dict[str, dict[int, list]] = {}
    for pair_key, n_data in pairs.items():
        for n_str, metric_data in n_data.items():
            N = int(n_str)
            for metric, stats in metric_data.items():
                val = stats.get(field, float("nan"))
                if val is None or not np.isfinite(val):
                    continue
                aggregated.setdefault(metric, {}).setdefault(N, []).append(val)

    # Operating-budget points (full-pool bootstrap), if present.
    operating: dict[str, dict[str, list]] = {}
    op_data = data.get("operating", {}) if isinstance(data, dict) else {}
    for pair_key, entry in op_data.items():
        op_N = entry.get("N")
        for metric, stats in entry.get("stats", {}).items():
            val = stats.get(field, float("nan"))
            if val is None or not np.isfinite(val):
                continue
            d = operating.setdefault(metric, {"N": [], "values": []})
            d["N"].append(op_N)
            d["values"].append(val)

    return aggregated, operating


def plot_sensitivity(aggregated: dict, ax: plt.Axes, title: str, field: str,
                     tolerance: float | None, ymax: float | None):
    """Curve mode: bootstrap sample-efficiency curves (no operating-point stars)."""
    for metric, label in METRICS:
        if metric not in aggregated:
            continue
        n_vals = sorted(aggregated[metric].keys())
        means = np.array([np.mean(aggregated[metric][N]) for N in n_vals])
        stds = np.array([np.std(aggregated[metric][N]) for N in n_vals])

        color = METRIC_COLORS.get(metric, "grey")
        ls = METRIC_LINESTYLE.get(metric, "-")
        ax.plot(n_vals, means, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.8, label=label, zorder=3)
        ax.fill_between(n_vals, means - stds, means + stds, color=color,
                        alpha=0.10, zorder=1)

    if tolerance is not None:
        ax.axhline(tolerance, color="black", linewidth=0.8, linestyle=":")

    ax.set_xscale("log")
    any_metric = next((m for m, _ in METRICS if m in aggregated), None)
    if any_metric is not None:
        ax.set_xticks(sorted(aggregated[any_metric].keys()))
        ax.set_xticks([], minor=True)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0, top=ymax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"evaluation trajectories $N$", fontsize=9)
    ylab = (r"std / signal range" if field == "std_over_range"
            else "CV (std / mean)")
    ax.set_ylabel(f"{ylab}\n(lower = more stable)", fontsize=9)
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="x", which="major", linestyle=":", linewidth=0.5, alpha=0.6)


def summarize_operating(operating: dict) -> dict:
    """{metric: {'N':[], 'values':[]}} -> {metric: {'mean','std','N'}} across pairs."""
    out = {}
    for metric, d in operating.items():
        vals = [v for v in d["values"] if np.isfinite(v)]
        if not vals:
            continue
        out[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "N": int(np.median(d["N"])) if d["N"] else None,
        }
    return out


def plot_operating_bars(env_summaries, ax, field, tolerance):
    """
    Operating mode: grouped bar chart of the full-pool bootstrap precision.
    Groups = environments, bars within a group = metrics. One estimator only,
    so it is not conflated with the bootstrap curve.

    env_summaries: list of (label, summary, op_N).
    """
    metric_keys = [m for m, _ in METRICS]
    metric_labels = dict(METRICS)
    n_env = len(env_summaries)
    n_metrics = len(metric_keys)
    bar_w = 0.8 / n_metrics

    for gi, (label, summary, op_N) in enumerate(env_summaries):
        for mi, metric in enumerate(metric_keys):
            if metric not in summary:
                continue
            x = gi + (mi - (n_metrics - 1) / 2.0) * bar_w
            color = METRIC_COLORS.get(metric, "grey")
            ax.bar(x, summary[metric]["mean"], width=bar_w * 0.92, color=color,
                   yerr=summary[metric]["std"], capsize=2,
                   error_kw={"elinewidth": 0.8},
                   label=metric_labels[metric] if gi == 0 else None)

    if tolerance is not None:
        ax.axhline(tolerance, color="black", linewidth=1.0, linestyle=":",
                   label="tolerance", zorder=0)

    ax.set_xticks(range(n_env))
    ax.set_xticklabels([f"{label}\n(N={op_N})" for label, _, op_N in env_summaries])
    ylab = (r"std / signal range" if field == "std_over_range"
            else "CV (std / mean)")
    ax.set_ylabel(f"{ylab}  (lower = more stable)", fontsize=10)
    ax.set_title("Estimation precision at the operating budget (full-pool bootstrap)",
                 fontsize=12)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, ncol=n_metrics + 1, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))

    # headroom so the tolerance line and tallest bar are both visible
    tops = [summary[m]["mean"] + summary[m]["std"]
            for _, summary, _ in env_summaries for m in summary]
    top = max(tops + ([tolerance] if tolerance else [0.0]))
    ax.set_ylim(0, top * 1.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One JSON file per environment")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display name for each input file (same order)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path. Defaults to "
                             "figures/n_sensitivity_curve.png (curve mode) or "
                             "figures/n_sensitivity_operating.png (operating mode).")
    parser.add_argument("--mode", choices=["curve", "operating"], default="curve",
                        help="curve: bootstrap sample-efficiency curves (no stars). "
                             "operating: grouped bar chart of full-pool bootstrap "
                             "precision at the operating budget.")
    parser.add_argument("--field", type=str, default="std_over_range",
                        choices=["std_over_range", "cv"],
                        help="Which stability field to plot")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Reference line for 'stable enough' "
                             "(set to a negative value to omit)")
    parser.add_argument("--ymax", type=float, default=0.3,
                        help="Curve mode only: upper y-limit to clip the large "
                             "small-N bands (set <=0 for auto)")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("--inputs and --labels must have the same number of entries")
        sys.exit(1)

    if args.output is None:
        args.output = (f"figures/n_sensitivity_{args.mode}.png")

    tol = args.tolerance if args.tolerance >= 0 else None

    if args.mode == "curve":
        ymax = args.ymax if args.ymax > 0 else None
        n = len(args.inputs)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, path, label in zip(axes, args.inputs, args.labels):
            aggregated, _ = load_and_aggregate(path, args.field)
            plot_sensitivity(aggregated, ax, label, args.field, tol, ymax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=9, loc="lower center",
                   ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("Metric estimation stability vs number of evaluation "
                     "trajectories N", fontsize=12, y=1.01)
    else:  # operating
        env_summaries = []
        for path, label in zip(args.inputs, args.labels):
            _, operating = load_and_aggregate(path, args.field)
            summ = summarize_operating(operating)
            op_N = next((summ[m]["N"] for m in summ), None)
            env_summaries.append((label, summ, op_N))
        fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 2.4 * len(env_summaries)), 4.5))
        plot_operating_bars(env_summaries, ax, args.field, tol)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

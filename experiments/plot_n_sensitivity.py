"""
Plot N sensitivity results produced by run_n_sensitivity.py.

Each input JSON corresponds to one environment/algorithm run. The script
aggregates a chosen stability field across all checkpoint pairs within each
file, then plots mean +/- std per metric as a function of N (the number of
evaluation trajectories).

The primary field is `std_over_range` (std normalized by the metric's signal
range across the checkpoint series): lower = more stable, and it is directly
comparable across metrics and robust to means that are near or cross zero. The
claim the figure supports is that the behavioral metrics drop below a given
stability tolerance at a *smaller* N than mean return does.

Usage:
    python experiments/plot_n_sensitivity.py \
        --inputs results/n_sensitivity_frozenlake.json \
                 results/n_sensitivity_mountaincar.json \
                 results/n_sensitivity_taxi.json \
        --labels FrozenLake MountainCar Taxi \
        --output figures/n_sensitivity.png \
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
    ("seff",                    r"$\mathcal{S}_{eff}$"),
    ("topological_shift",       r"$\Delta$Topo"),
    ("strategic_shift",         r"$\Delta$Strat"),
    ("3-gram_wasserstein",      r"$W_3$"),
    ("mean_return",             "Reward"),
]

METRIC_COLORS = {
    "seff":                "#9C27B0",   # purple (matches perplexity panels)
    "topological_shift":   "#2196F3",   # blue
    "strategic_shift":     "#FF9800",   # orange
    "3-gram_wasserstein":  "#4CAF50",   # green
    "mean_return":         "#E91E63",   # pink/red — reward baseline
}

METRIC_LINESTYLE = {
    "mean_return": "--",                 # dashed to visually separate reward
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
    return aggregated


def plot_sensitivity(aggregated: dict, ax: plt.Axes, title: str, field: str,
                     tolerance: float | None):
    for metric, label in METRICS:
        if metric not in aggregated:
            continue
        n_vals = sorted(aggregated[metric].keys())
        means = np.array([np.mean(aggregated[metric][N]) for N in n_vals])
        stds = np.array([np.std(aggregated[metric][N]) for N in n_vals])

        color = METRIC_COLORS.get(metric, "grey")
        ls = METRIC_LINESTYLE.get(metric, "-")
        ax.plot(n_vals, means, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.8, label=label)
        ax.fill_between(n_vals, means - stds, means + stds, color=color, alpha=0.15)

    if tolerance is not None:
        ax.axhline(tolerance, color="black", linewidth=0.8, linestyle=":")

    ax.set_xscale("log")
    # x ticks inferred from any metric present
    any_metric = next((m for m, _ in METRICS if m in aggregated), None)
    if any_metric is not None:
        ax.set_xticks(sorted(aggregated[any_metric].keys()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("N (evaluation trajectories)", fontsize=9)
    ylab = (r"std / signal range" if field == "std_over_range"
            else "CV (std / mean)")
    ax.set_ylabel(f"{ylab}\n(lower = more stable)", fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One JSON file per environment")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display name for each input file (same order)")
    parser.add_argument("--output", type=str, default="figures/n_sensitivity.png")
    parser.add_argument("--field", type=str, default="std_over_range",
                        choices=["std_over_range", "cv"],
                        help="Which stability field to plot")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Horizontal reference line for 'stable enough' "
                             "(set to a negative value to omit)")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("--inputs and --labels must have the same number of entries")
        sys.exit(1)

    tol = args.tolerance if args.tolerance >= 0 else None

    n = len(args.inputs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, path, label in zip(axes, args.inputs, args.labels):
        aggregated = load_and_aggregate(path, args.field)
        plot_sensitivity(aggregated, ax, label, args.field, tol)

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=8, loc="upper right")

    fig.suptitle("Metric estimation stability vs number of evaluation trajectories N",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

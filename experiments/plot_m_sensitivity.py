"""
Plot M sensitivity results produced by run_m_sensitivity.py.

Each input JSON corresponds to one environment/algorithm run.
The script aggregates r(M) across all checkpoint pairs within each file,
then plots mean ± std per metric as a function of M.

Usage:
    python experiments/plot_m_sensitivity.py \
        --inputs results/m_sensitivity_frozenlake.json \
                 results/m_sensitivity_mountaincar.json \
                 results/m_sensitivity_taxi.json \
        --labels FrozenLake MountainCar Taxi \
        --output figures/m_sensitivity.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

METRICS = [
    ("topological_shift",       r"$\Delta$Topo"),
    ("strategic_shift",         r"$\Delta$Strat"),
    ("3-gram_wasserstein",      r"$\Delta$Seq"),
    ("mean_return_diff",        "Reward"),
]

METRIC_COLORS = {
    "topological_shift":   "#2196F3",   # blue
    "strategic_shift":     "#FF9800",   # orange
    "3-gram_wasserstein":  "#4CAF50",   # green
    "mean_return_diff":    "#E91E63",   # pink/red  — reward baseline
}

METRIC_LINESTYLE = {
    "mean_return_diff": "--",           # dashed to visually separate reward
}


def load_and_aggregate(path: str) -> dict[str, dict[int, dict]]:
    """
    Load a sensitivity JSON and aggregate r(M) across all checkpoint pairs.

    Returns:
        {metric: {M: {"values": [r values across pairs]}}}
    """
    with open(path) as f:
        data = json.load(f)

    aggregated: dict[str, dict[int, list]] = {}

    for pair_key, metric_data in data.items():
        for metric, m_dict in metric_data.items():
            if metric not in aggregated:
                aggregated[metric] = {}
            for m_str, vals in m_dict.items():
                M = int(m_str)
                if M not in aggregated[metric]:
                    aggregated[metric][M] = []
                aggregated[metric][M].append(vals["mean_normalized"])

    return aggregated


def plot_sensitivity(aggregated: dict, ax: plt.Axes, title: str):
    for metric, label in METRICS:
        if metric not in aggregated:
            continue
        m_vals = sorted(aggregated[metric].keys())
        means, stds = [], []
        for M in m_vals:
            r_values = aggregated[metric][M]
            means.append(np.mean(r_values))
            stds.append(np.std(r_values))

        means = np.array(means)
        stds = np.array(stds)
        color = METRIC_COLORS.get(metric, "grey")
        ls = METRIC_LINESTYLE.get(metric, "-")

        ax.plot(m_vals, means, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.8, label=label)
        ax.fill_between(m_vals, means - stds, means + stds,
                        color=color, alpha=0.15)

    # reference line and operating point
    ax.axhline(0.9, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(20, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.2, 0.1))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("M (bootstrap samples)", fontsize=9)
    ax.set_ylabel(r"$r(M) = \varepsilon(M) / \varepsilon(M_{\max})$", fontsize=9)
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="x", which="major", linestyle=":", linewidth=0.5, alpha=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One JSON file per environment")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display name for each input file (same order)")
    parser.add_argument("--output", type=str, default="figures/m_sensitivity.png")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("--inputs and --labels must have the same number of entries")
        sys.exit(1)

    n = len(args.inputs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, path, label in zip(axes, args.inputs, args.labels):
        aggregated = load_and_aggregate(path)
        plot_sensitivity(aggregated, ax, label)

    # shared legend on the last axes
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=8, loc="lower right")

    # annotation for the operating point
    for ax in axes:
        ax.annotate("M=20", xy=(20, 0.02), fontsize=7, ha="center", color="grey")

    fig.suptitle("Noise threshold stability vs bootstrap sample count M",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

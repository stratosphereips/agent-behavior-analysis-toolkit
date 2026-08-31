"""Behavioral-fingerprint / sequential-checkpoint diagnostics (the current method).

`plot_sequential_cp_metrics` is the dense developer diagnostic (5-panel stacked
figure) emitted by `experiments/sequential_cp_comparison.py`. The shipped per-run
report figure lives in `utils/plot_fingerprint_report.py`, not here.
"""
import numpy as np
import matplotlib.pyplot as plt


def _plot_noise_markers(ax, x, y, noise_hi, noise_lo=None, **line_kwargs):
    """
    Plot a line with 'x' markers where the metric exceeds the noise estimate and
    'o' markers where it does not.  For a two-sided band pass noise_lo as well.
    Returns the line's resolved color string.
    """
    x_arr = np.array(x)
    y_arr = np.array(y, dtype=float)
    hi_arr = np.array(noise_hi, dtype=float)
    valid = ~np.isnan(y_arr)
    if noise_lo is not None:
        lo_arr = np.array(noise_lo, dtype=float)
        above = valid & ((y_arr > hi_arr) | (y_arr < lo_arr))
    else:
        above = valid & (y_arr > hi_arr)
    at_or_below = valid & ~above
    line = ax.plot(x_arr, y_arr, **line_kwargs)[0]
    c = line.get_color()
    if above.any():
        ax.scatter(x_arr[above], y_arr[above], marker='x', color=c, zorder=5, s=50)
    if at_or_below.any():
        ax.scatter(x_arr[at_or_below], y_arr[at_or_below], marker='o', color=c, zorder=5, s=20)
    return c


def plot_sequential_cp_metrics(checkpoint_labels, metrics, run_name):
    """
    Generates the 5-panel stacked metrics figure for sequential checkpoint comparison.

    Args:
        checkpoint_labels: list of int checkpoint ids (x-axis values)
        metrics: dict with keys matching those built in sequential_cp_comparison.main()
        run_name: string used in the figure title

    Returns:
        matplotlib Figure
    """
    x_deltas = checkpoint_labels[1:]

    fig, axes = plt.subplots(5, 1, figsize=(14.4, 20), sharex=True)
    fig.suptitle(f"Run: {run_name}", fontsize=20)

    # Plot 1: Reward
    axes[0].plot(checkpoint_labels, metrics["mean_return"], label="Mean Return", marker='x')
    axes[0].fill_between(
        checkpoint_labels,
        np.array(metrics["mean_return"]) - np.array(metrics["std_return"]),
        np.array(metrics["mean_return"]) + np.array(metrics["std_return"]),
        alpha=0.3,
    )

    # r_true (true/underlying reward) is optional: only some trajectories track it
    # (via info["r_true"]). Only overlay it when at least one checkpoint has it.
    mean_r_true = metrics.get("mean_r_true")
    if mean_r_true and any(v is not None for v in mean_r_true):
        r_true_mean = np.array([v if v is not None else np.nan for v in mean_r_true], dtype=float)
        r_true_std = np.array([v if v is not None else np.nan for v in metrics.get("std_r_true", [])], dtype=float)
        axes[0].plot(checkpoint_labels, r_true_mean, label="Mean r_true", marker='o', color='orange')
        axes[0].fill_between(
            checkpoint_labels,
            r_true_mean - r_true_std,
            r_true_mean + r_true_std,
            alpha=0.3,
            color='orange',
        )

    axes[0].set_ylabel("Return")
    axes[0].set_title("Model Performance")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)

    # Plot 2: Perplexity
    axes[1].plot(checkpoint_labels, metrics["state_visitation_perplexity"], label="State Visitation Perplexity", marker='x', color='purple')
    axes[1].plot(checkpoint_labels, metrics["total_nodes"], label="Visited Nodes (= max perplexity)", linestyle='--', color='gray', alpha=0.7)
    axes[1].set_ylabel("Number of States")
    axes[1].set_title("State Visitation Perplexity")
    max_nodes = max(metrics["total_nodes"]) if metrics["total_nodes"] else 1
    axes[1].set_ylim(0, max_nodes * 1.05)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True)

    # Plot 3: Shifts (with noise estimates as dashed lines in matching colors)
    color_topo = _plot_noise_markers(axes[2], x_deltas, metrics["topological_shift_raw"],
                                     metrics["topological_shift_noise_threshold"],
                                     label="Full Topological Shift")
    # overlap is a decomposition channel: its noise floor is pruned by the producer.
    # plot the raw line with floor markers only when the floor is present.
    overlap_floor = metrics.get("topological_shift_overlap_noise_threshold")
    if overlap_floor is not None:
        color_topo_overlap = _plot_noise_markers(axes[2], x_deltas, metrics["topological_shift_overlap_raw"],
                                                 overlap_floor,
                                                 label="Topological Shift on Overlap")
    else:
        color_topo_overlap = axes[2].plot(x_deltas, metrics["topological_shift_overlap_raw"],
                                          label="Topological Shift on Overlap")[0].get_color()
    strategic_arr = np.array(metrics["strategic_shift_raw"], dtype=float)
    color_strategic = _plot_noise_markers(axes[2], x_deltas, strategic_arr,
                                          metrics["strategic_shift_noise_threshold"],
                                          label="Strategic Shift")
    nan_mask = np.isnan(strategic_arr)
    if nan_mask.any():
        axes[2].scatter(np.array(x_deltas)[nan_mask], np.ones(nan_mask.sum()) * 1.0,
                        marker='D', color='red', zorder=5, s=60,
                        label="Strategic Shift: no shared states")
    axes[2].plot(x_deltas, metrics["topological_shift_noise_threshold"], linestyle='--', color=color_topo, alpha=0.7, label="Full Topological Shift noise estimate")
    if overlap_floor is not None:
        axes[2].plot(x_deltas, overlap_floor, linestyle='--', color=color_topo_overlap, alpha=0.7, label="Topological Shift on Overlap noise estimate")
    axes[2].plot(x_deltas, metrics["strategic_shift_noise_threshold"], linestyle='--', color=color_strategic, alpha=0.7, label="Strategic Shift noise estimate")
    axes[2].set_ylabel("JSD [0, 1]")
    axes[2].set_title("Behavioral Shifts")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].grid(True)

    # Plot 4: Topological Turnover (directional non-overlap: discovery vs abandonment)
    # Discovery is drawn upward and abandonment downward from a zero baseline, so the
    # individual magnitudes are visible; the net flux (discovery - abandonment) is
    # overlaid as a line. This distinguishes a frozen footprint (both bars near zero)
    # from a relocating one (both bars large), cases that the net line alone conflates.
    have_dir = (
        "topological_shift_discovery_raw" in metrics
        and "topological_shift_abandonment_raw" in metrics
        and len(metrics["topological_shift_discovery_raw"]) > 0
    )
    if have_dir:
        disc = np.array(metrics["topological_shift_discovery_raw"])
        aban = np.array(metrics["topological_shift_abandonment_raw"])

        # bar width from the x spacing (training steps); fall back to 1.0
        if len(x_deltas) >= 2:
            bar_width = 0.4 * (x_deltas[1] - x_deltas[0])
        else:
            bar_width = 1.0

        axes[3].axhline(0.0, color='gray', linewidth=1.0, alpha=0.7)

        # net two-sided noise band around zero
        if "topological_shift_net_noise_hi" in metrics and "topological_shift_net_noise_lo" in metrics:
            net_hi = np.array(metrics["topological_shift_net_noise_hi"])
            net_lo = np.array(metrics["topological_shift_net_noise_lo"])
            axes[3].fill_between(x_deltas, net_lo, net_hi, color='gray', alpha=0.2,
                                 label="Net flux noise band (two-sided)")

        # discovery upward (green), abandonment downward (red)
        axes[3].bar(x_deltas, disc, width=bar_width, color='green', alpha=0.6,
                    label="Discovery (added-node mass)")
        axes[3].bar(x_deltas, -aban, width=bar_width, color='red', alpha=0.6,
                    label="Abandonment (removed-node mass)")

        # per-component one-sided noise floors (discovery above zero, abandonment mirrored below)
        if "topological_shift_discovery_noise_threshold" in metrics:
            disc_floor = np.array(metrics["topological_shift_discovery_noise_threshold"])
            axes[3].plot(x_deltas, disc_floor, linestyle='--', color='green', alpha=0.7,
                         label="Discovery noise floor")
        if "topological_shift_abandonment_noise_threshold" in metrics:
            aban_floor = np.array(metrics["topological_shift_abandonment_noise_threshold"])
            axes[3].plot(x_deltas, -aban_floor, linestyle='--', color='red', alpha=0.7,
                         label="Abandonment noise floor")

        # net flux line overlaid (= discovery - abandonment)
        if "topological_shift_net_raw" in metrics:
            net = np.array(metrics["topological_shift_net_raw"])
            if "topological_shift_net_noise_hi" in metrics and "topological_shift_net_noise_lo" in metrics:
                _plot_noise_markers(axes[3], x_deltas, net,
                                    metrics["topological_shift_net_noise_hi"],
                                    noise_lo=metrics["topological_shift_net_noise_lo"],
                                    color='black', label="Net flux (discovery - abandonment)")
            else:
                axes[3].plot(x_deltas, net, marker='x', color='black',
                             label="Net flux (discovery - abandonment)")

    axes[3].set_ylabel("JSD (+ discovery, − abandonment)")
    axes[3].set_title("Topological Turnover (discovery up, abandonment down)")
    axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[3].grid(True)

    # Plot 5: Ratios and Distances (with noise estimates as dashed lines)
    w_dist = np.array(metrics["3-gram_wasserstein_raw"])
    w_noise = np.array(metrics["3-gram_wasserstein_noise_threshold"])
    max_w = 3  # max distance between two 3-grams
    color_wass = _plot_noise_markers(axes[4], x_deltas, w_dist / max_w, w_noise / max_w,
                                     label="Norm. Wasserstein 3-gram")
    axes[4].plot(x_deltas, w_noise / max_w, linestyle='--', color=color_wass, alpha=0.7, label="Norm. Wasserstein 3-gram noise estimate")

    overlap_arr = np.array(metrics["node_overlap"])
    visited_arr = np.array(metrics["total_nodes"][1:])
    axes[4].plot(x_deltas, overlap_arr / np.maximum(visited_arr, 1), label="Overlap / Visited Nodes", marker='x')

    perp_arr = np.array(metrics["state_visitation_perplexity"])
    axes[4].plot(checkpoint_labels, perp_arr / np.maximum(np.array(metrics["total_nodes"]), 1), label="Perplexity / Visited Nodes", marker='x')

    axes[4].set_xlabel("Training Step")
    axes[4].set_ylabel("Ratio [0, 1]")
    axes[4].set_title("Normalized Distances and Ratios")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_xticks(checkpoint_labels)
    axes[4].tick_params(axis='x', rotation=45)
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[4].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

"""Per-trajectory visualizations: action-usage-over-time and segment-cluster features.

Used by the replay / generalization experiments to inspect the raw behavior of a
single checkpoint's evaluation trajectories.
"""
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def plot_segment_cluster_features(clusters: dict) -> plt.Figure:
    feature_names = ["λ_ret", "λ_ret_std", "surprise", "surprise_std",
                     "reward", "reward_std", "length", "pos_start", "pos_end"]
    #feature_names = ["λ_ret", "surprise", "surprise_std",
    #                 "reward", "reward_std", "length", "pos_start", "pos_end"]
    feature_names = ["λ_ret", "surprise", "surprise_std",
        "reward", "reward_std", "length", "pos_start", "pos_end", "state_diversity", "action_diversity"]

    cluster_data = {}
    for cluster_id, segments in clusters.items():
        avg_features = {}
        std_features = {}
        for feature_idx, feature in enumerate(feature_names):
            values = [seg["features"][feature_idx] for seg in segments]
            avg_features[feature] = np.mean(values)
            std_features[feature] = np.std(values)
        cluster_data[cluster_id] = {"avg": avg_features, "std": std_features}

    n_clusters = len(cluster_data)
    n_features = len(feature_names)
    bar_width = 0.8 / n_clusters
    x = np.arange(n_features)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20.colors

    for i, cluster_id in enumerate(sorted(cluster_data)):
        values = [cluster_data[cluster_id]["avg"][f] for f in feature_names]
        stds = [cluster_data[cluster_id]["std"][f] for f in feature_names]
        offset = x - 0.4 + i * bar_width + bar_width/2
        ax.bar(offset, values, yerr=stds, width=bar_width, color=colors[i % len(colors)], label=f"Cluster {cluster_id}")

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel("Average Value (symlog scale)")
    ax.set_title("Average Segment Feature Values per Cluster")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig


def plot_action_per_step_distribution(
    trajectories: Iterable, global_actions: list, normalize=True, dpi=600, title="Action Distribution per Time Step"
) -> plt.Figure:
    """
    Plots a stacked bar chart of the distribution of actions taken at each time step
    across multiple trajectories. Also accounts for how many trajectories survive
    to each step.

    Parameters:
        trajectories: iterable of trajectories, each trajectory is a list of transitions
                      where transition.action is an int in [0, num_actions-1]
        num_actions: number of discrete actions
        normalize: if True, show proportions instead of counts
        dpi: figure DPI (default 600)
    """

    action_to_idx = {}
    max_len = max([len(trajectory) for trajectory in trajectories])

    # Action counts and trajectory survival counts
    # Action counts and trajectory survival counts
    if isinstance(global_actions, int):
        num_actions = global_actions
        global_actions_list = list(range(num_actions))
    else:
        num_actions = len(global_actions)
        global_actions_list = global_actions

    action_counts = np.zeros((max_len, num_actions), dtype=float)
    traj_counts = np.zeros(max_len, dtype=int)  # number of trajectories that reached step i
    action_to_idx = {action: idx for idx, action in enumerate(global_actions_list)}

    for trajectory in trajectories:
        for i, transition in enumerate(trajectory):
            traj_counts[i] += 1
            if not isinstance(transition.action, int):
                # If action is not int, try to find it in map
                if transition.action in action_to_idx:
                    action_idx = action_to_idx[transition.action]
                else:
                    # Fallback or error? Let's skip or warn.
                    # For now, if we can't map it, we might crash.
                    continue
            else:
                # If action is int, checks range
                if 0 <= transition.action < num_actions:
                    action_idx = transition.action
                else:
                     continue

            action_counts[i, action_idx] += 1

    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            action_counts = np.divide(
                action_counts,
                traj_counts[:, None],
                where=traj_counts[:, None] > 0
            )

    # Plot stacked bar chart
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=dpi)
    bottom = np.zeros(max_len)
    for action in global_actions_list:
         action_idx = action_to_idx[action]
         if action_idx not in action_to_idx.values():
             action_name = f"Action {action_idx}"
         else:
             action_name = action
         ax1.bar(
             np.arange(max_len),
             action_counts[:, action_idx],
             bottom=bottom,
             label=action_name,
         )
         bottom += action_counts[:, action_idx]

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Ratio" if normalize else "Count")
    ax1.set_title(title)

    # --- Plot trajectory survival line (only once, normalized to [0,1]) ---
    traj_counts_norm = traj_counts / np.max(traj_counts) if np.max(traj_counts) > 0 else traj_counts
    line, = ax1.plot(
        np.arange(max_len),
        traj_counts_norm,
        color="black", linestyle="--", label="# trajectories"
    )

    # --- Right axis shows counts corresponding to the same line ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of trajectories")
    ax2.set_ylim(0, np.max(traj_counts))  # raw counts on right axis

    # Merge legends (only once)
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=min(3, len(labels1)))

    plt.tight_layout()
    return fig

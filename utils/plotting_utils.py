
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from typing import Iterable

from sklearn.base import defaultdict
from utils.trajectory_utils import compute_trajectory_surprises, find_trajectory_segments

def plot_trajectory_segments(trajectories:Iterable, policy, previous_policy, filename)->plt.Figure:
    max_len = max([len(t) for t in trajectories])
    surprise_matrix = np.full((len(trajectories), max_len), np.nan)
    for i, trajectory in enumerate(trajectories):
        surprises = compute_trajectory_surprises(trajectory, policy, previous_policy)
        for j in range(0, len(surprises)):
            surprise_matrix[i,j] = surprises[j]
    fig, ax = plt.subplots(figsize=(10, 5))
    norm = mcolors.SymLogNorm(linthresh=10, linscale=1, vmin=-500, vmax=500)
    cmap = plt.cm.seismic
    cmap_with_grey = cmap.copy()
    cmap_with_grey.set_bad(color='lightgrey')
    im = ax.imshow(surprise_matrix, cmap=cmap_with_grey, interpolation='none', aspect='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Surprise')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Trajectory')
    ax.set_title('Surprise Across Trajectories (SymLogNorm)')
    return fig

def plot_segment_cluster_features(clusters:dict)->plt.Figure:
    cluster_data = {}
    for cluster_id, segments in clusters.items():
        feature_sums = {}
        for segment in segments:
            for feature, value in segment["features"].items():
                feature_sums[feature] = feature_sums.get(feature, 0) + value
        cluster_data[cluster_id] = {}
        for feature_idx, feature_name in enumerate(sorted(feature_sums)):
            # average for all segments in the cluster
            v = feature_sums[feature_name]/len(segments)
            cluster_data[cluster_id][feature_idx] = v
    x = np.arange(len(feature_sums))
    bar_width = 0.2
    group_spacing = 0.3
    group_width = len(clusters) * bar_width + group_spacing
    x = np.arange(len(feature_sums)) * group_width
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2.colors  # or any other color palette
    for i, cluster in enumerate(cluster_data):
        values = list(cluster_data[cluster].values())
        offset_x = x + i * bar_width
        ax.bar(offset_x, values, width=bar_width, label=f"Cluster {cluster}", color=colors[i % len(colors)])
    # Formatting
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(sorted(feature_sums), rotation=45)
    ax.set_ylabel("Value")
    ax.set_title("Feature values per cluster")
    plt.yscale('log') 
    ax.legend()
    plt.tight_layout()
    return fig

def plot_action_per_step_distribution(trajectories: Iterable, num_actions: int, normalize=True) -> plt.Figure:
    """
    Plots a stacked bar chart of the distribution of actions taken at each time step
    across multiple trajectories.

    Parameters:
        trajectories: iterable of trajectories, each trajectory is a list of transitions
                      where transition.action is an int in [0, num_actions-1]
        num_actions: number of discrete actions
        normalize: if True, show proportions instead of counts
    """
    # Count actions per timestep
    max_len = max(len(trajectory) for trajectory in trajectories)
    action_counts = np.zeros((max_len, num_actions), dtype=float)
    for trajectory in trajectories:
        for i, transition in enumerate(trajectory):
            action_counts[i, transition.action] += 1

    if normalize:
        action_counts = action_counts / action_counts.sum(axis=1, keepdims=True)

    # Plot stacked bars
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(max_len)

    for action_idx in range(num_actions):
        ax.bar(
            np.arange(max_len),
            action_counts[:, action_idx],
            bottom=bottom,
            label=f"Action {action_idx}"
        )
        bottom += action_counts[:, action_idx]

    ax.set_xlabel("Time step")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title("Action Distribution per Time Step")
    ax.legend(title="Actions")
    plt.tight_layout()
    return fig
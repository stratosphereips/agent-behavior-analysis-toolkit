
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Iterable
from utils.trajectory_utils import compute_trajectory_surprises, find_trajectory_segments

def plot_trajectory_segments(trajectories:Iterable, policy, previous_policy, filename)->plt.Figure:
    max_len = max([len(t) for t in trajectories])
    surprise_matrix = np.full((len(trajectories), max_len), np.nan)
    highlight_segments = {}
    for i, trajectory in enumerate(trajectories):
        surprises = compute_trajectory_surprises(trajectory, policy, previous_policy)
        #highlight_segments[i] = find_trajectory_segments(trajectory, previous_policy)
        for j in range(0, len(surprises)):
            surprise_matrix[i,j] = surprises[j]
    fig, ax = plt.subplots(figsize=(10, 5))
    print(np.mean(surprise_matrix,axis=0))
    cmap = plt.cm.viridis
    cmap_with_grey = cmap.copy()
    cmap_with_grey.set_bad(color='lightgrey')
    im = ax.imshow(surprise_matrix, cmap=cmap_with_grey, interpolation='none', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Surprise')
    # Highlight steps using rectangles
    # for row_idx, segments in highlight_segments.items():
    #     for segment in segments:
    #         start = segment["start"]
    #         end = segment["end"]
    #         width = end - start + 1
    #         rect = Rectangle((start - 0.5, row_idx - 0.5), width, 1,
    #              linewidth=1.5, edgecolor='black',
    #              facecolor='none', hatch='///',  alpha=0.2)
    #         ax.add_patch(rect)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Trajectory')
    ax.set_title('Surprise Across Trajectories')
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
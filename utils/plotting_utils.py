from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from typing import Iterable
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from sklearn.base import defaultdict
from trajectory import Trajectory
from utils.trajectory_utils import compute_trajectory_surprises, find_trajectory_segments

def plot_trajectory_surprise_matrix(surprise_matrix: np.ndarray) -> plt.Figure:
    """
    Plots a heatmap of surprise values across trajectories.

    Parameters:
        surprise_matrix: 2D numpy array of shape (num_trajectories, max_len)
                         containing surprise values (np.nan for missing).

    Returns:
        fig: matplotlib Figure object
    """
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

def plot_segment_cluster_features(clusters: dict) -> plt.Figure:
    feature_names = ["λ_ret", "λ_ret_std", "surprise", "surprise_std", 
                     "reward", "reward_std", "length", "pos_start", "pos_end"]
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
    trajectories: Iterable, num_actions: int, normalize=True, dpi=600
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
    action_counts = np.zeros((max_len, num_actions), dtype=float)
    traj_counts = np.zeros(max_len, dtype=int)  # number of trajectories that reached step i

    for trajectory in trajectories:
        for i, transition in enumerate(trajectory):
            traj_counts[i] += 1
            if not isinstance(transition.action, int):
                if transition.action.type not in action_to_idx:
                    action_to_idx[transition.action.type] = len(action_to_idx)
                action_idx = action_to_idx[transition.action.type]
            else:
                action_idx = transition.action
            action_counts[i, action_idx] += 1

    if normalize:
        # Normalize by number of trajectories that reached each step
        with np.errstate(invalid="ignore", divide="ignore"):
            action_counts = np.divide(
                action_counts,
                traj_counts[:, None],
                where=traj_counts[:, None] > 0
            )

    # Plot stacked bars
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=dpi)
    bottom = np.zeros(max_len)

    for action_idx in range(num_actions):
        if action_idx not in action_to_idx.values():
            action_name = f"Action {action_idx}"
        else:
            # recover action name from mapping
            action_name = list(action_to_idx.keys())[list(action_to_idx.values()).index(action_idx)]
        ax1.bar(
            np.arange(max_len),
            action_counts[:, action_idx],
            bottom=bottom,
            label=action_name,
        )
        bottom += action_counts[:, action_idx]

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Proportion" if normalize else "Count")
    ax1.set_title("Action Distribution per Time Step")

    # Plot trajectory survival as a line on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(np.arange(max_len), traj_counts, color="black", linestyle="--", label="# trajectories")
    ax2.set_ylabel("Number of trajectories")

    # Merge legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, title="Legend", loc="upper right")

    plt.tight_layout()
    return fig

def plot_quantile_fan(data, num_quantiles=5, title="Surprise distribution per step", dpi=600, figsize=(10,6)):
    """
    Plot quantile fan chart over steps.

    Parameters
    ----------
    data : np.ndarray
        Matrix of shape (num_trajectories, max_steps).
    num_quantiles : int
        Number of quantiles to compute (e.g. 5 → 0%,25%,50%,75%,100%).
    title : str
        Plot title.
    dpi : int
        Dots per inch (controls resolution).
    figsize : tuple
        Size of the figure in inches (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    num_trajectories, max_steps = data.shape
    steps = np.arange(max_steps)

    # Define quantiles
    quantiles = np.linspace(0, 100, num_quantiles)
    q_values = np.nanpercentile(data, quantiles, axis=0)  # shape: (num_quantiles, max_steps)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot median
    median = q_values[num_quantiles // 2]
    ax.plot(steps, median, color="black", label="Median (50%)", linewidth=1.2)

    # Shade bands between symmetric quantiles
    for i in range(num_quantiles // 2):
        lower = q_values[i]
        upper = q_values[-(i+1)]
        alpha = 0.2 + 0.1 * i  # darker toward median
        ax.fill_between(steps, lower, upper, alpha=alpha, label=f"{quantiles[i]}–{quantiles[-(i+1)]}%")

    ax.set_xlabel("Step")
    ax.set_ylabel("Surprise")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_trajectory_network_colored_nodes_by_cluster(trajectory: Trajectory, segments: dict[int, list[tuple[int, int]]]):
    # G = nx.DiGraph()
    
    # # Build nodes and edges
    # for i, transition in enumerate(trajectory):
    #     G.add_node(transition.state)
    #     G.add_node(transition.next_state)
    #     G.add_edge(transition.state, transition.next_state,
    #                action=transition.action, reward=transition.reward)
    
    # # Map states to cluster IDs
    # state_to_cluster = {}
    # print(f"Trajectory len={len(trajectory)}")
    # for cluster_id, seg_list in segments.items():
    #     print(f"Cluster {cluster_id} segments:")
    #     for seg in seg_list:
    #         print(f"\t{seg['start']} - {seg['end']}")
    #         start = seg["start"]
    #         end = seg["end"]
    #         for step in range(start, end):
    #             state_to_cluster[trajectory[step].state] = cluster_id
    #         # Optionally include the next_state of the last step
    #         #state_to_cluster[trajectory[end].next_state] = cluster_id
    
    # # Color map for clusters
    # cluster_ids = sorted(segments.keys())
    # cmap = plt.cm.get_cmap('Set1', len(cluster_ids))
    # cluster_to_color = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}
    
    # node_colors = []
    # for state in G.nodes():
    #     cid = state_to_cluster.get(state, None)
    #     node_colors.append(cluster_to_color[cid] if cid is not None else "lightgray")
    
    # plt.figure(figsize=(20, 6))
    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args='-Grankdir=LR -Granksep=3 -Gnodesep=2')

    # nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, connectionstyle="arc3,rad=0.5")

    # edge_labels = {(u, v): d["action"] for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(
    #     G, pos,
    #     edge_labels=edge_labels,
    #     label_pos=0.5,
    #     rotate=True,
    #     font_size=7,
    #     bbox=None
    # )

    # nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=node_colors, edgecolors="black")
    # nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    # # Legend handles
    # handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}', markerfacecolor=cluster_to_color[cid], markersize=5) for cid in cluster_ids]
    # handles.append(plt.Line2D([0], [0], marker='o', color='w', label='No cluster', markerfacecolor='lightgrey', markersize=5))
    # plt.legend(
    #     handles=handles,
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, -0.15),
    #     ncol=min(6, len(handles)),  # wrap legend if too many clusters
    #     handleheight=1.5,
    #     handlelength=1.5,
    #     fontsize=8
    # )
    # plt.axis('off')
    # plt.title("Trajectory Network with Node Colors by Cluster")
    # plt.savefig("Testgraph.png", dpi=500)
    G = nx.DiGraph()
    node_labels = {}

    # Create unique nodes per timestep
    for i, transition in enumerate(trajectory):
        curr_node = f"{transition.state}"
        next_node = f"{transition.next_state}"
       # If edge exists, append the new label
        if G.has_edge(curr_node, next_node):
            old_label = G[curr_node][next_node].get('action', '')
            new_label = (f"{transition.action}@{old_label.split('@')[-1]}, {i}") if old_label else f"{transition.action}@{i}"
            G[curr_node][next_node]['action'] = new_label
        else:
            G.add_edge(curr_node, next_node, action=f"{transition.action}@{i}")
        node_labels[curr_node] = str(transition.state)
        node_labels[next_node] = str(transition.next_state)
    def rgba_to_hex(rgba):
        r, g, b = [int(255*x) for x in rgba[:3]]  # ignore alpha
        return f"#{r:02x}{g:02x}{b:02x}"
    # Cluster coloring
    node_to_cluster = {}
    cluster_ids = sorted(segments.keys())
    cmap = plt.cm.Set1.colors
    cluster_to_color = {cid: rgba_to_hex(cmap[i]) for i, cid in enumerate(cluster_ids)}

    for cluster_id, seg_list in segments.items():
        for segment in seg_list:
            start = segment["start"]
            end = segment["end"]
            for step in range(start, end):
                node_to_cluster[f"{trajectory[step].state}"] = cluster_id
            # node_to_cluster[f"{trajectory[end].next_state}@{end+1}"] = cluster_id
    
    node_colors = []
    nodes_sorted = list(G.nodes())
    for node in nodes_sorted:
        cid = node_to_cluster.get(node, None)
        color = cluster_to_color[cid] if cid is not None else "lightgray"
        node_colors.append(color)

    # Sort nodes by their first appearance in the trajectory    # Convert to AGraph
    A = to_agraph(G)
    A.graph_attr.update(rankdir='LR', nodesep='1.5', ranksep='2')  # spacing

    # Customize nodes
    for i, node in enumerate(nodes_sorted):
        n = A.get_node(node)
        n.attr['style'] = 'dotted' if i == 0 else 'solid'
        n.attr['shape'] = 'square' if i == len(nodes_sorted) - 1 else 'circle'
        n.attr['fillcolor'] = node_colors[i]
        n.attr['style'] += ',filled'
        n.attr['color'] = 'black'

    # Customize edges
    for u, v, d in G.edges(data=True):
        e = A.get_edge(u, v)
        e.attr['label'] = str(d['action'])
        e.attr['fontsize'] = '10'
        e.attr['fontcolor'] = 'black'
        if u == v:
            e.attr['dir'] = 'forward'
            e.attr['arrowhead'] = 'normal'
            e.attr['minlen'] = '2'
            e.attr['constraint'] = 'false'
        else:
            e.attr['arrowhead'] = 'normal'

    # Layout and draw
    A.layout(prog='dot')
    png_bytes = A.draw(format='png')
    return png_bytes

import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_trajectory_heatmap(surprise, action_change, cluster, gap=1, min_height=4, max_height=25):
    """
    Plots a heatmap for trajectory data: surprise, action change, cluster.
    
    Args:
        surprise: 2D array of surprise values
        action_change: 2D array of 0/1 action change flags
        cluster: 2D array of cluster IDs (-1 = outlier)
        gap: vertical gap between bands
        min_height: min figure height
        max_height: max figure height
    """
    n_traj, n_steps = surprise.shape
    band_height = 3
    total_band = band_height + gap

    # --- Stack bands with gaps ---
    stacked = np.full((n_traj * total_band, n_steps), np.nan)
    for t in range(n_traj):
        base = t * total_band
        stacked[base + 0] = surprise[t]
        stacked[base + 1] = action_change[t]
        stacked[base + 2] = cluster[t]

    # --- Figure size ---
    fig_height = np.clip(n_traj * 0.05 * total_band, min_height, max_height) + 1.2
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=400)

    # --- Colormaps ---
    cmap_surprise = plt.cm.seismic
    norm_surprise = mcolors.SymLogNorm(linthresh=5, linscale=1, vmin=-100, vmax=100)

    cmap_binary = mcolors.ListedColormap(["white", "black"])
    norm_binary = mcolors.BoundaryNorm([0, 0.5, 1], 2)

    mask = np.isnan(stacked)

    # --- Helper to plot each band ---
    def plot_masked_rows(mod, cmap, norm=None):
        msk = mask.copy()
        for i in range(stacked.shape[0]):
            if i % total_band != mod:
                msk[i] = True
        ax.imshow(np.ma.masked_where(msk, stacked), aspect="auto", cmap=cmap, norm=norm)

    # Plot surprise and action change
    plot_masked_rows(0, cmap_surprise, norm_surprise)
    plot_masked_rows(1, cmap_binary, norm_binary)

    # --- Plot clusters ---
    cluster_mask = mask.copy()
    for i in range(stacked.shape[0]):
        if i % total_band != 2:
            cluster_mask[i] = True

    # Masked cluster array (still 2D)
    masked_clusters = np.ma.masked_where(cluster_mask, stacked)

    # Map unique clusters to colors
    unique_clusters = np.unique(cluster[~np.isnan(cluster)])
    set1_colors = plt.cm.Set1.colors
    cluster_colors = {}
    color_idx = 0
    for c in sorted(unique_clusters):
        if c == -1:
            cluster_colors[c] = (0.8, 0.8, 0.8, 1.0)  # light gray for outlier
        else:
            cluster_colors[c] = set1_colors[color_idx % len(set1_colors)]
            color_idx += 1

    # Create ListedColormap and bounds for imshow
    cmap_list = [cluster_colors[c] for c in sorted(cluster_colors)]
    cmap_cluster = mcolors.ListedColormap(cmap_list)
    bounds = sorted(cluster_colors)
    norm_cluster = mcolors.BoundaryNorm(bounds + [bounds[-1]+1], cmap_cluster.N)

    ax.imshow(masked_clusters, aspect="auto", cmap=cmap_cluster, norm=norm_cluster)

    # --- Axis labels ---
    ytick_every = max(1, n_traj // 20)
    yticks = [(t * total_band) + 1.5 for t in range(0, n_traj, ytick_every)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"T{t}" for t in range(0, n_traj, ytick_every)], fontsize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Trajectory")

    # --- Surprise colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap_surprise, norm=norm_surprise)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Surprise", fontsize=8)

    # --- Legends above the figure ---
    cluster_handles = [
        mpatches.Patch(color=color, label=("No cluster" if c==-1 else f"C{c}"))
        for c, color in cluster_colors.items()
    ]
    cluster_legend = ax.legend(
        handles=cluster_handles, title="Clusters",
        loc='lower left', bbox_to_anchor=(0, 1.05),
        ncol=min(5, len(cluster_handles)), fontsize=6
    )
    ax.add_artist(cluster_legend)

    binary_handles = [
        mpatches.Patch(color="white", edgecolor="black", label="Same action"),
        mpatches.Patch(color="black", label="Action changed")
    ]
    binary_legend = ax.legend(
        handles=binary_handles, title="Action Change",
        loc='lower left', bbox_to_anchor=(0.5, 1.025), fontsize=6
    )
    ax.add_artist(binary_legend)

    plt.tight_layout(rect=[0, 0, 0.95, 0.92])
    return fig, ax


def plot_cluster_distribution_per_step(clusters, trajectory_len, normalize=True, dpi=600):
    """
    Plot a stacked bar chart of the proportion of segments from each cluster at each time step.
    
    Parameters:
        clusters: dict(cluster_id -> list of segments), each segment has 'pos_start' and 'pos_end'
        trajectory_len: maximum trajectory length (number of steps)
        normalize: if True, show proportions instead of counts
        dpi: figure DPI
    """
    # Count number of segments per cluster at each step
    cluster_ids = sorted(clusters.keys())
    step_counts = np.zeros((trajectory_len, len(cluster_ids)), dtype=float)

    cluster_idx_map = {cid: i for i, cid in enumerate(cluster_ids)}

    for cid, seg_list in clusters.items():
        for seg in seg_list:
            start = seg["features"][-2]  # pos_start
            end = seg["features"][-1]  # pos_end
            # Increment count for each step the segment spans
            step_counts[start:end, cluster_idx_map[cid]] += 1

    if normalize:
        # Normalize per step to get proportions
        totals = step_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        totals[totals == 0] = 1
        step_counts = step_counts / totals

    # Use tab20 colors for clusters
    tab20_colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    bottom = np.zeros(trajectory_len)

    for i, cid in enumerate(cluster_ids):
        color = tab20_colors[i % len(tab20_colors)]
        ax.bar(np.arange(trajectory_len), step_counts[:, i], bottom=bottom, label=f"Cluster {cid}", color=color)
        bottom += step_counts[:, i]

    ax.set_xlabel("Time step")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title("Cluster Distribution per Time Step")
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return fig

def _dedup_segments_by_features(segments):
    """Keep only one segment per unique feature tuple signature."""
    seen, uniq = set(), []
    for s in segments:
        sig = s["features"]  # Use the precomputed tuple
        if sig not in seen:
            seen.add(sig)
            uniq.append(s)
    return uniq


def visualize_clusters(
    clusters,
    max_trajectory_len: int,
    default_feature_name="surprises",
    heatmap_cmap="seismic",
    dpi=600,
):
    """
    Heatmap of per-step surprise with left strip showing cluster membership.
    - Deduplicate segments by features
    - Segments ordered by cluster, then by start index, then by trajectory_id (if present)
    - Surprise values sym-log normalized
    """
    rows, row_cluster = [], []

    # collect rows
    for cid in sorted(clusters):
        segments = _dedup_segments_by_features(clusters[cid])
        # Sort by start, then trajectory_id if present
        def seg_sort_key(s):
            tid = s.get("trajectory_id", 0)
            return (s["start"], tid)
        for seg in sorted(segments, key=seg_sort_key):
            start, end = seg["start"], seg["end"]
            vals = np.asarray(seg[default_feature_name])
            row = np.full(max_trajectory_len, np.nan)
            row[start:start + (end - start)] = vals[: end - start]
            rows.append(row)
            row_cluster.append(cid)

    if not rows:
        raise ValueError("No segments to plot after deduplication.")

    heatmap = np.vstack(rows)
    cmap = plt.cm.get_cmap(heatmap_cmap) if isinstance(heatmap_cmap, str) else heatmap_cmap
    cmap_with_grey = cmap.copy()
    cmap_with_grey.set_bad(color='lightgrey')
    # cluster strip
    unique_cids = list(dict.fromkeys(row_cluster))  # preserve cluster order
    cid_to_idx = {c: i for i, c in enumerate(unique_cids)}
    strip = np.array([cid_to_idx[c] for c in row_cluster])[:, None]  # (rows, 1)

    # cluster colors (tab20 cycles automatically)
    tab20 = plt.cm.get_cmap("tab20").colors
    colors = [tab20[i % len(tab20)] for i in range(len(unique_cids))]
    cmap_clusters = mcolors.ListedColormap(colors)
    norm_clusters = mcolors.BoundaryNorm(
        np.arange(-0.5, len(unique_cids) + 0.5, 1), cmap_clusters.N
    )

    # sym-log normalization
    norm_surprise = mcolors.SymLogNorm(linthresh=3, linscale=2, vmin=-100, vmax=100)

    # ---- plot ----
    fig = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.035, 0.965])
    ax_strip, ax_heat = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)

    # strip
    ax_strip.imshow(strip, aspect="auto", cmap=cmap_clusters, norm=norm_clusters)
    ax_strip.axis("off")

    # heatmap
    im = ax_heat.imshow(np.ma.masked_invalid(heatmap),
                        aspect="auto", cmap=cmap_with_grey, norm=norm_surprise)
    fig.colorbar(im, ax=ax_heat, pad=0.01).set_label(default_feature_name.capitalize())
    ax_heat.set(xlabel="Trajectory Step", ylabel="Unique Segments (rows)")

    # legend
    handles = [mpatches.Patch(color=colors[i], label=f"Cluster {c}") 
               for i, c in enumerate(unique_cids)]
    ax_heat.legend(handles=handles, title="Clusters",
                   loc="upper left", bbox_to_anchor=(1.02, 1.0))

    return fig
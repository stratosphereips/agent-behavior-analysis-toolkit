import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Iterable
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from sklearn.base import defaultdict
from trajectory import Trajectory

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
    # for action_idx in range(num_actions):
    #     if action_idx not in action_to_idx.values():
    #         action_name = f"Action {action_idx}"
    #     else:
    #         action_name = list(action_to_idx.keys())[list(action_to_idx.values()).index(action_idx)]
    #     ax1.bar(
    #         np.arange(max_len),
    #         action_counts[:, action_idx],
    #         bottom=bottom,
    #         label=action_name,
    #     )
    #     bottom += action_counts[:, action_idx]

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
    
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Step")
    ax.set_ylabel("Surprise (symlog scale)")
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
    #     node_colors.append(cluster_to_color[cid] if cid is not None else "lightgrey")
    
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
            start = seg["features"][-4]  # pos_start
            end = seg["features"][-3]  # pos_end
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
    return segments


def visualize_clusters(
    clusters,
    max_trajectory_len: int,
    default_feature_name="surprises",
    heatmap_cmap="seismic",
    dpi=500,
):
    """
    Heatmap of per-step surprise with left strips showing cluster membership and segment return.
    - Deduplicate segments by features
    - Segments ordered by cluster, then by start index, then by trajectory_id (if present)
    - Surprise values sym-log normalized
    - Cluster strip (tab20), Return strip (inferno)
    """
    rows, row_cluster, row_return = [], [], []

    # collect rows
    for cid in sorted(clusters):
        segments = _dedup_segments_by_features(clusters[cid])
        def seg_sort_key(s):
            tid = s.get("trajectory_id", 0)
            return (s["start"], tid)
            #return (tid, s["start"], s["end"])
        for seg in sorted(segments, key=seg_sort_key):
            start, end = seg["start"], seg["end"]
            vals = np.asarray(seg[default_feature_name])
            row = np.full(max_trajectory_len, np.nan)
            row[start:start + (end - start)] = vals[: end - start]
            rows.append(row)
            row_cluster.append(cid)
            row_return.append(seg["return"])

    if not rows:
        raise ValueError("No segments to plot after deduplication.")

    heatmap = np.vstack(rows)
    # Cluster strip
    unique_cids = list(dict.fromkeys(row_cluster))  # preserve cluster order
    cid_to_idx = {c: i for i, c in enumerate(unique_cids)}
    strip_cluster = np.array([cid_to_idx[c] for c in row_cluster])[:, None]  # (rows, 1)

    # Return strip
    returns = np.array(row_return)
    vmin, vmax = np.nanmin(returns), np.nanmax(returns)
    norm_return = mcolors.Normalize(vmin=vmin, vmax=vmax)
    strip_return = returns[:, None]  # (rows, 1)

    # Colormaps
    cmap = plt.cm.get_cmap(heatmap_cmap) if isinstance(heatmap_cmap, str) else heatmap_cmap
    cmap_with_grey = cmap.copy()
    cmap_with_grey.set_bad(color='lightgrey')
    tab20 = plt.cm.get_cmap("tab20").colors
    colors = [tab20[i % len(tab20)] for i in range(len(unique_cids))]
    cmap_clusters = mcolors.ListedColormap(colors)
    norm_clusters = mcolors.BoundaryNorm(
        np.arange(-0.5, len(unique_cids) + 0.5, 1), cmap_clusters.N
    )
    cmap_return = plt.cm.inferno

    norm_surprise = mcolors.SymLogNorm(linthresh=10, linscale=1, vmin=-500, vmax=500)
    # ---- plot ----
    fig = plt.figure(figsize=(13, 7), constrained_layout=True, dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.035, 0.93, 0.035], wspace=0.15)
    ax_strip, ax_heat, ax_return = (
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1], sharey=None),
        fig.add_subplot(gs[2], sharey=None)
    )

    # Cluster strip (left)
    ax_strip.imshow(strip_cluster, aspect="auto", cmap=cmap_clusters, norm=norm_clusters)
    ax_strip.axis("off")

    # Heatmap (center)
    im = ax_heat.imshow(np.ma.masked_invalid(heatmap),
                        aspect="auto", cmap=cmap_with_grey, norm=norm_surprise)
    cbar_surprise = fig.colorbar(im, ax=ax_heat, pad=0.01)
    cbar_surprise.set_label(default_feature_name.capitalize())
    ax_heat.set(xlabel="Trajectory Step", ylabel="Unique Segments (rows)")

    # Return strip (right)
    im_return = ax_return.imshow(strip_return, aspect="auto", cmap=cmap_return, norm=norm_return)
    ax_return.axis("off")
    # Make return colorbar the same height as surprise colorbar
    cbar_return = fig.colorbar(im_return, ax=ax_return, pad=0.15)
    cbar_return.set_label("Mean Segment Return", labelpad=20)
    # Match the height of the return colorbar to the surprise colorbar
    cbar_return.ax.set_position(cbar_surprise.ax.get_position())

    # legend below the heatmap
    handles = [mpatches.Patch(color=colors[i], label=f"Cluster {c}") 
               for i, c in enumerate(unique_cids)]
    ncol = len(handles) if len(handles) <= 10 else 2
    ax_heat.legend(
        handles=handles,
        title="Clusters",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=ncol,
        fontsize=10
    )

    return fig


# --- Helper function for the first plot ---
def plot_action_per_step_distribution_helper(
    trajectories: Iterable, num_actions: int, ax: plt.Axes, normalize=True
):
    """Plots the action distribution onto a given axis, including the trajectory count on a twin axis."""
    
    # 1. Data Preparation
    action_to_idx = {}
    max_len = max([len(trajectory) for trajectory in trajectories])
    action_counts = np.zeros((max_len, num_actions), dtype=float)
    traj_counts = np.zeros(max_len, dtype=int)

    for trajectory in trajectories:
        for i, transition in enumerate(trajectory):
            traj_counts[i] += 1
            
            # Action index handling (keeping original logic)
            if not isinstance(transition.action, int):
                if hasattr(transition.action, 'type'):
                    if transition.action.type not in action_to_idx:
                        action_to_idx[transition.action.type] = len(action_to_idx)
                    action_idx = action_to_idx[transition.action.type]
                else:
                    # Fallback if transition.action is not an int and has no .type
                    action_idx = 0 # Or handle error appropriately
            else:
                action_idx = transition.action
                
            if 0 <= action_idx < num_actions:
                action_counts[i, action_idx] += 1
            else:
                # Handle actions outside expected range if necessary
                pass

    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            action_counts = np.divide(
                action_counts,
                traj_counts[:, None],
                where=traj_counts[:, None] > 0
            )

    # 2. Plot Stacked Bars (on the main axis)
    bottom = np.zeros(max_len)
    
    # Determine action names
    action_names = {}
    for i in range(num_actions):
        if i not in action_to_idx.values():
            action_names[i] = f"Action {i}"
        else:
            name = list(action_to_idx.keys())[list(action_to_idx.values()).index(i)]
            action_names[i] = name
            
    for action_idx in range(num_actions):
        ax.bar(
            np.arange(max_len),
            action_counts[:, action_idx],
            bottom=bottom,
            label=action_names.get(action_idx, f"Action {action_idx}"),
            width=1.0 # Use width=1.0 for standard stacked bar chart look
        )
        bottom += action_counts[:, action_idx]

    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title("Action Distribution per Time Step")
    
    # 3. Plot trajectory survival line (on the twin axis)
    ax_twin = ax.twinx()
    
    max_traj_count = np.max(traj_counts)
    if max_traj_count > 0:
        traj_counts_norm = traj_counts / max_traj_count
    else:
        traj_counts_norm = traj_counts # all zeros

    line, = ax_twin.plot(
        np.arange(max_len),
        traj_counts_norm,
        color="black", linestyle="--", label="# trajectories"
    )

    ax_twin.set_ylabel("Number of trajectories (Right Axis)")
    ax_twin.set_ylim(0, max_traj_count)
    
    # 4. Merge Legends
    handles1, labels1 = ax.get_legend_handles_labels()
    # The line handle is on the twin axis
    ax.legend(handles1 + [line], labels1 + ["# trajectories"], title="Actions", loc="upper right")
    
    # Remove x-axis labels/ticks since it's not the bottom plot
    ax.tick_params(labelbottom=False)
    ax_twin.tick_params(labelbottom=False)
    
    return ax, ax_twin


# --- Helper function for the second plot ---
def plot_cluster_distribution_per_step_helper(clusters: dict, trajectory_len: int, ax: plt.Axes, normalize=True):
    """Plots the cluster distribution onto a given axis."""
    
    # 1. Data Preparation
    cluster_ids = sorted(clusters.keys())
    step_counts = np.zeros((trajectory_len, len(cluster_ids)), dtype=float)
    cluster_idx_map = {cid: i for i, cid in enumerate(cluster_ids)}

    for cid, seg_list in clusters.items():
        for seg in seg_list:
            # Assuming 'features' is indexed correctly as per original function
            if 'features' in seg and len(seg['features']) >= 4:
                start = int(seg["features"][-4])
                end = int(seg["features"][-3])
                # Increment count for each step the segment spans
                step_counts[start:end, cluster_idx_map[cid]] += 1
            else:
                # Handle error or skip segment
                pass

    if normalize:
        # Normalize per step to get proportions
        totals = step_counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        step_counts = step_counts / totals

    # 2. Plot Stacked Bars
    tab20_colors = plt.cm.tab20.colors
    bottom = np.zeros(trajectory_len)

    for i, cid in enumerate(cluster_ids):
        color = tab20_colors[i % len(tab20_colors)]
        ax.bar(np.arange(trajectory_len), step_counts[:, i], bottom=bottom, label=f"Cluster {cid}", color=color, width=1.0)
        bottom += step_counts[:, i]

    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title("Cluster Distribution per Time Step")
    # Place legend outside to not overlap with the plot
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove x-axis labels/ticks since it's not the bottom plot
    ax.tick_params(labelbottom=False)
    
    return ax


# --- Helper function for the third plot ---
def plot_trajectory_surprise_matrix_helper(surprise_matrix: np.ndarray, ax: plt.Axes):
    """Plots a heatmap of surprise values onto a given axis."""
    
    # 1. Setup Heatmap
    norm = mcolors.SymLogNorm(linthresh=10, linscale=1, vmin=-500, vmax=500)
    cmap = plt.cm.seismic
    cmap_with_grey = cmap.copy()
    cmap_with_grey.set_bad(color='lightgrey')
    
    im = ax.imshow(surprise_matrix, cmap=cmap_with_grey, interpolation='none', aspect='auto', norm=norm)
    
    # 2. Set Labels and Title
    ax.set_xlabel('Time Step') # Keep this label only on the bottom plot
    ax.set_ylabel('Trajectory')
    ax.set_title('Surprise Across Trajectories (SymLogNorm)')
    
    # 3. Add Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Surprise')
    
    return ax


# =========================================================================
# === The Main Concatenation Function ===
# =========================================================================

def plot_combined_trajectory_analysis(
    trajectories: Iterable, 
    num_actions: int, 
    clusters: dict, 
    trajectory_len: int, 
    surprise_matrix: np.ndarray,
    dpi: int = 600,
    figsize: tuple = (10, 15)
) -> plt.Figure:
    """
    Concatenates three trajectory-analysis plots vertically with a shared, aligned X-axis.
    
    Parameters:
        trajectories: Data for action distribution plot.
        num_actions: Number of discrete actions.
        clusters: Data for cluster distribution plot.
        trajectory_len: Maximum length of trajectories (for cluster plot x-limit).
        surprise_matrix: 2D array for the surprise heatmap.
        dpi: Figure DPI.
        figsize: Figure size (width, height).
        
    Returns:
        The combined matplotlib Figure object.
    """
    
    # 1. Setup the figure and shared axes
    fig, axes = plt.subplots(
        nrows=3, ncols=1, 
        sharex=True, # **This is the critical part for axis alignment**
        figsize=figsize, 
        dpi=dpi,
        # Increase vertical space between plots to accommodate titles/labels
        gridspec_kw={'hspace': 0.4} 
    )
    ax_action, ax_cluster, ax_surprise = axes

    # 2. Plot Action Distribution (Top Plot)
    plot_action_per_step_distribution_helper(
        trajectories, num_actions, ax_action, normalize=True
    )

    # 3. Plot Cluster Distribution (Middle Plot)
    plot_cluster_distribution_per_step_helper(
        clusters, trajectory_len, ax_cluster, normalize=True
    )
    
    # 4. Plot Surprise Matrix (Bottom Plot)
    plot_trajectory_surprise_matrix_helper(
        surprise_matrix, ax_surprise
    )
    
    # 5. Final Polish
    # Adjust the X-axis limits for all plots based on the longest data source
    max_x_len = max(len(trajectory) for trajectory in trajectories)
    ax_action.set_xlim(0, max_x_len)
    
    plt.tight_layout()
    # Adjust layout to make room for legends/colorbars outside the main column
    plt.subplots_adjust(right=0.8) 

    return fig




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
    color_topo = axes[2].plot(x_deltas, metrics["topological_shift_raw"], label="Full Topological Shift", marker='x')[0].get_color()
    color_topo_overlap = axes[2].plot(x_deltas, metrics["topological_shift_overlap_raw"], label="Topological Shift on Overlap", marker='x')[0].get_color()
    strategic_arr = np.array(metrics["strategic_shift_raw"], dtype=float)
    color_strategic = axes[2].plot(x_deltas, strategic_arr, label="Strategic Shift", marker='x')[0].get_color()
    nan_mask = np.isnan(strategic_arr)
    if nan_mask.any():
        axes[2].scatter(np.array(x_deltas)[nan_mask], np.ones(nan_mask.sum()) * 1.0,
                        marker='D', color='red', zorder=5, s=60,
                        label="Strategic Shift: no shared states")
    axes[2].plot(x_deltas, metrics["topological_shift_noise_threshold"], linestyle='--', color=color_topo, alpha=0.7, label="Full Topological Shift noise estimate")
    axes[2].plot(x_deltas, metrics["topological_shift_overlap_noise_threshold"], linestyle='--', color=color_topo_overlap, alpha=0.7, label="Topological Shift on Overlap noise estimate")
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
    color_wass = axes[4].plot(x_deltas, w_dist / max_w, label="Norm. Wasserstein 3-gram", marker='x')[0].get_color()
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


def plot_behavioral_ontogeny(checkpoint_labels, metrics, every_nth=1):
    """
    Generates the 3-Row Behavioral Ontogeny Dashboard.
    
    Args:
        checkpoint_labels (list): Strings or Ints for the X-axis (e.g. ['100', '200'...])
        metrics (dict): Dictionary containing the keys defined in your setup.
        every_nth (int): Plot only ever Nth checkpoint (subsampling).
    """
    
    # 1. Setup Data & Canvas
    # Convert labels to integers for plotting if possible, else use range
    try:
        x_full = np.array([int(lbl) for lbl in checkpoint_labels])
    except ValueError:
        x_full = np.arange(len(checkpoint_labels))
        
    # --- SUBSAMPLING LOGIC ---
    # We must handle the fact that some metrics (shifts) might be Length N-1.
    # To simplify, we first pad everything to Length N, then subsample.
    
    def ensure_length_n(data, n):
        arr = np.array(data)
        if len(arr) == n - 1:
            return np.concatenate(([0], arr))
        return arr

    # Prepare FULL LENGTH arrays first
    n_full = len(x_full)
    
    # keys that are explicitly N length in typical usage
    # We'll just grab everything as arrays first.
    
    # Subsample indices
    indices = np.arange(0, n_full, every_nth)
    x = x_full[indices]
        
    has_surprise = False
    nrows = 4 if has_surprise else 3
    figsize = (12, 18) if has_surprise else (12, 14)
        
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
    if has_surprise:
        ax_perf, ax_state, ax_dyn, ax_surp = axes
    else:
        ax_perf, ax_state, ax_dyn = axes
    
    # Grid settings for professional look
    grid_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.4}

    # =========================================================================
    # ROW 1: PERFORMANCE (Reward)
    # Goal: "Is the agent winning?"
    # =========================================================================
    
    r_mean = np.array(metrics['reward'])[indices]
    r_std = np.array(metrics['reward_std'])[indices]
    
    # Plot Mean
    ax_perf.plot(x, r_mean, color='black', linewidth=2.5, marker='o', label='Mean Reward')
    
    # Plot Standard Deviation Shading
    ax_perf.fill_between(x, r_mean - r_std, r_mean + r_std, color='gray', alpha=0.25, label='Std Dev')
    
    ax_perf.set_title("A. Performance (External Evaluation)", loc='left', fontweight='bold', fontsize=12)
    ax_perf.set_ylabel("Reward")
    ax_perf.grid(**grid_style)
    ax_perf.legend(loc="upper left", frameon=True)

    # =========================================================================
    # ROW 2: BEHAVIORAL STATE (Volume, Depth, Confidence)
    # Goal: "How is the agent behaving?" (Structure & Stability)
    # =========================================================================
    
    # --- LEFT AXIS: Strategic Confidence (0.0 - 1.0) ---
    conf = np.array(metrics['empirical_policy_certainty'])[indices]
    conf_noise = np.array(metrics['empirical_policy_certainty_noise'])[indices]
    
    # Clip error bars to [0, 1]
    c_lower = np.clip(conf - conf_noise, 0, 1)
    c_upper = np.clip(conf + conf_noise, 0, 1)
    
    ax_state.plot(x, conf, color='tab:blue', linewidth=2, marker='s', label='Policy Certainty ($C_{\pi}$)')
    ax_state.fill_between(x, c_lower, c_upper, color='tab:blue', alpha=0.2)
    
    ax_state.set_ylabel("Policy Certainty ($C_{\pi}$)", color='tab:blue', fontweight='bold')
    ax_state.tick_params(axis='y', labelcolor='tab:blue')
    ax_state.set_ylim(-0.05, 1.05) # Slight buffer
    ax_state.set_title("B. Behavioral Structure", loc='left', fontweight='bold', fontsize=12)
    ax_state.grid(**grid_style)

    # --- RIGHT AXIS: Magnitude Metrics (Log Scale) ---
    ax_state2 = ax_state.twinx()
    ax_state2.set_yscale('log')
    
    # 1. Exploration Volume (Red)
    vol = np.array(metrics['effective_state_coverage'])[indices]
    vol_noise = np.array(metrics['effective_state_coverage_noise'])[indices]
    
    # Safety: Ensure lower bound is at least 1 (log(0) crash prevention)
    v_lower = np.maximum(vol - vol_noise, 1.0)
    v_upper = vol + vol_noise
    
    ax_state2.plot(x, vol, color='tab:red', linestyle='--', linewidth=2, marker='^', label='Effective State Coverage')
    ax_state2.fill_between(x, v_lower, v_upper, color='tab:red', alpha=0.15)
    
    # 2. Traversal Depth (Green)
    # Logic: Plot Reliable Min as line. Shade up to Max (Min + Noise).
    depth_min = np.array(metrics['robust_traversal_depth'])[indices]
    depth_noise = np.array(metrics['robust_traversal_depth_noise'])[indices]
    depth_max = depth_min + depth_noise
    
    ax_state2.plot(x, depth_min, color='tab:green', linestyle=':', linewidth=2.5, marker='D', label='Robust Traversal Depth')
    ax_state2.fill_between(x, depth_min, depth_max, color='tab:green', alpha=0.2) # The "Luck Gap"

    ax_state2.set_ylabel("Coverage / Depth (Log Scale)", color='black', fontweight='bold')
    
    # Combined Legend for Row 2 (Tricky with twin axes)
    lines_L, labels_L = ax_state.get_legend_handles_labels()
    lines_R, labels_R = ax_state2.get_legend_handles_labels()
    ax_state.legend(lines_L + lines_R, labels_L + labels_R, loc='best', frameon=True, ncol=2)


    # =========================================================================
    # ROW 3: DYNAMICS (Shifts)
    # Goal: "Is the agent changing?" (Derivatives)
    # =========================================================================
    
    # Helper to align Deltas (Length N-1) with Checkpoints (Length N)
    # Then subsample
    
    topo_full = ensure_length_n(metrics['topological_shift'], n_full)
    strat_full = ensure_length_n(metrics['strategic_shift'], n_full)
    
    topo = topo_full[indices]
    strat = strat_full[indices]
    
    # Plot De-noised Signals
    ax_dyn.plot(x, topo, color='purple', linewidth=2, marker='v', label='Topological Shift ($\Delta_{Topo}$)')
    ax_dyn.fill_between(x, 0, topo, color='purple', alpha=0.1) # Highlight events
    
    ax_dyn.plot(x, strat, color='orange', linewidth=2, marker='x', label='Strategic Shift ($\Delta_{Strat}$)')
    ax_dyn.fill_between(x, 0, strat, color='orange', alpha=0.1) # Highlight events
    
    ax_dyn.set_title("C. Policy Dynamics", loc='left', fontweight='bold', fontsize=12)
    ax_dyn.set_ylabel("Shift Magnitude (JSD)")
    ax_dyn.set_xlabel("Training Checkpoints")
    ax_dyn.grid(**grid_style)
    ax_dyn.legend(loc="upper left", frameon=True)
    
    # Refine Layout
    plt.tight_layout()
    
    return fig

def plot_behavioral_ontogeny_multiseed(checkpoint_labels, multiseed_metrics: list[dict], every_nth=1):
    """
    Generates a Behavioral Ontogeny Dashboard for multiple seeds.
    Each metric is plotted in its own subplot to prevent overcrowding.
    
    Args:
        checkpoint_labels (list): Strings or Ints for the X-axis (e.g. ['100', '200'...])
        multiseed_metrics (list of dict): List of dictionaries, each containing metrics for one seed.
        every_nth (int): Plot only ever Nth checkpoint (subsampling).
    """
    # 1. Setup Data & Canvas
    try:
        x_full = np.array([int(lbl) for lbl in checkpoint_labels])
    except ValueError:
        x_full = np.arange(len(checkpoint_labels))
        
    def ensure_length_n(data, n):
        arr = np.array(data)
        if len(arr) == n - 1:
            return np.concatenate(([0], arr))
        return arr

    n_full = len(x_full)
    indices = np.arange(0, n_full, every_nth)
    x = x_full[indices]
    
    # We plot 6 metrics explicitly
    nrows = 6
    figsize = (12, 18)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
    ax_rew, ax_conf, ax_cov, ax_depth, ax_topo, ax_strat = axes
    
    grid_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.4}

    ax_rew.set_title("Mean Return", loc='left', fontweight='bold', fontsize=12)
    ax_rew.set_ylabel("Mean Return")
    
    ax_conf.set_title("Policy Certainty ($C_{\pi}$)", loc='left', fontweight='bold', fontsize=12)
    ax_conf.set_ylabel("Certainty")
    ax_conf.set_ylim(-0.05, 1.05)
    
    ax_cov.set_title("Effective State Coverage (Log Scale)", loc='left', fontweight='bold', fontsize=12)
    ax_cov.set_ylabel("Coverage")
    ax_cov.set_yscale('log')
    
    ax_depth.set_title("Robust Traversal Depth (Log Scale)", loc='left', fontweight='bold', fontsize=12)
    ax_depth.set_ylabel("Depth")
    ax_depth.set_yscale('log')
    
    ax_topo.set_title("Topological Shift ($\Delta_{Topo}$)", loc='left', fontweight='bold', fontsize=12)
    ax_topo.set_ylabel("Shift (JSD)")
    
    ax_strat.set_title("Strategic Shift ($\Delta_{Strat}$)", loc='left', fontweight='bold', fontsize=12)
    ax_strat.set_ylabel("Shift (JSD)")
    ax_strat.set_xlabel("Training Checkpoints")

    for ax in axes:
        ax.grid(**grid_style)
        ax.tick_params(labelbottom=True)

    cmap = plt.get_cmap('tab10')
    
    for seed_idx, metrics in enumerate(multiseed_metrics):
        color = cmap(seed_idx % 10)
        label = f'Seed {seed_idx}'
        
        r_mean = np.array(metrics.get('reward', np.zeros(n_full)))[indices]
        ax_rew.plot(x, r_mean, color=color, linewidth=2, label=label)
        
        conf = np.array(metrics.get('empirical_policy_certainty', np.zeros(n_full)))[indices]
        ax_conf.plot(x, conf, color=color, linewidth=2, label=label)
        
        vol = np.array(metrics.get('effective_state_coverage', np.ones(n_full)))[indices]
        ax_cov.plot(x, vol, color=color, linewidth=2, label=label)
        
        depth_min = np.array(metrics.get('robust_traversal_depth', np.ones(n_full)))[indices]
        ax_depth.plot(x, depth_min, color=color, linewidth=2, label=label)
        
        topo_full = ensure_length_n(metrics.get('topological_shift', np.zeros(n_full-1)), n_full)
        ax_topo.plot(x, topo_full[indices], color=color, linewidth=2, label=label)
        
        strat_full = ensure_length_n(metrics.get('strategic_shift', np.zeros(n_full-1)), n_full)
        ax_strat.plot(x, strat_full[indices], color=color, linewidth=2, label=label)

    # Put legend outside the first plot
    ax_rew.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, title="Seeds")
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    return fig

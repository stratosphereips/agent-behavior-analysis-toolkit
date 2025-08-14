
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
    colors = plt.cm.Set1.colors  # or any other color palette
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
    # handles.append(plt.Line2D([0], [0], marker='o', color='w', label='No cluster', markerfacecolor='lightgray', markersize=5))
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
#from AIDojoCoordinator.utils.trajectory_analysis import read_json
import numpy as np
import os 
#import AIDojoCoordinator.utils.utils as utils
import argparse
import matplotlib.pyplot as plt
import pickle
#from scipy.stats import pearsonr
import pandas as pd
from collections import namedtuple
# import ot
# import ruptures as rpt
# from scipy.spatial.distance import mahalanobis
# from scipy.linalg import inv
# from ruptures.base import BaseCost
import networkx as nx
from typing import Hashable, Iterable

Transition = namedtuple("Trainsiton", ["state", "action", "reward", "next_state"])

def jaccard_similarity_coefficient(s1, s2):
    """Compute Jaccard similarity for two sets"""
    intersection = len(s1 & s2)  # Count overlapping directed edges
    union = len(s1) + len(s2) - intersection
    return intersection / union if union > 0 else 0

def overlap_coefficient(s1, s2):
    return len(s1 & s2)/min(len(s1), len(s2))

def graph_trainsion_cross_entropy(G1, G2):
    def cross_entropy(p, q, epsilon=1e-12):
        """
        Computes the cross-entropy between two probability distributions.

        Args:
            p: NumPy array representing the first probability distribution.
            q: NumPy array representing the second probability distribution.
            epsilon: Small value to avoid log(0).

        Returns:
            The cross-entropy between p and q.
        """

        q = np.clip(q, epsilon, 1 - epsilon)  # Clip to avoid log(0) and log(1)
        ce = -np.sum(p * np.log(q))
        return ce
    nodes = G1.node_set | G2.node_set
    node_ids = {n:i for i,n in enumerate(nodes)}
    probs1 = np.zeros([len(nodes), len(nodes)])
    for (s,a,d), prob in G1.get_probability_per_edge().items():
        probs1[node_ids[s], node_ids[d]] += prob
    probs2 = np.zeros([len(nodes), len(nodes)])
    for (s,a,d), prob in G2.get_probability_per_edge().items():
        probs2[node_ids[s], node_ids[d]] += prob
    row_ces = []
    for i in range(len(nodes)):
        row_ces.append(cross_entropy(probs1[i], probs2[i]))
    return np.mean(row_ces)

def plot_tg_mdp(graph, filename):
    """
    Plots the transition graph of a Markov Decision Process (MDP) using NetworkX.

    Args:
        graph: The transition graph of the MDP.
    """
    G = nx.MultiDiGraph()
    for (s1,s2,a), freq in graph.get_probability_per_edge().items():
        G.add_edge(s1, s2, weight=freq, action=a)

    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for u, v, k, d in G.edges(data=True, keys=True):
        label = f"{d['action']} ({d['weight']:.2f})"
        A.get_edge(u, v, k).attr['label'] = label

    A.graph_attr.update({
        "rankdir": "LR",     # Left to Right layout
        "nodesep": "0.2",    # ↓ Horizontal space between nodes (default is 0.25–0.5)
        "ranksep": "0.3",    # ↓ Vertical space between levels/ranks
        "splines": "true",
    }
    )
    # Set node colors
    for n in G.nodes():
        node = A.get_node(n)
        if n in graph.starting_states:
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = "green"
        elif n in graph.terminal_states:
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = "red"
        else:
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = 'white'  # or leave unstyled
        # plt.figure(figsize=(32, 24))
    # pos = nx.planar_layout(G)
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_labels(G, pos)
    # #     nx.draw(G, pos, with_labels=True, node_size=200, node_color="lightblue", edge_color="gray", arrowsize=15)
    # #     edge_labels = {(u, v): f"{data['action']} ({data['weight']:.2f})" for u, v, key, data in G.edges(data=True, keys=True)
    # # }   

    # nx.draw_networkx_edges(
    #     G, pos,
    #     edgelist=G.edges(keys=True),
    #     connectionstyle="arc3,rad=0.2",  # << makes curved edges!
    #     arrows=True
    # )
    # edge_labels = {(u, v): f"{data['action']} ({data['weight']:.2f})" for u, v, key, data in G.edges(data=True, keys=True)}

    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.6, font_size=10)
    # plt.title("Transition Graph of RL Agent")
    A.layout(prog='dot')
    A.draw(filename)

class TrajectoryGraph:
    """
    Class representing a graph induced by a set of trajectories. 
    """
    def __init__(self):
        # self._state2id = {}
        # self._id2state = {}
        # self._action2id = {}
        # self._id2action = {}
        self._edge_reward = {}
        self._edge_count = {}
        self._returns = []
        self._lengths = []
        self._starting_states = set()
        self._terminal_states = set()
    
    @property
    def num_trajectories(self)->int:
        return len(self._returns)
    
    # def get_state_id(self, state:Hashable)->int:
    #     if state not in self._state2id:
    #         self._state2id[state] = len(self._state2id)
    #         self._id2state[self._state2id[state]] = state
    #     return self._state2id[state]

    # def get_state(self, id:int)->Hashable:
    #     return self._id2state.get(id, None)
    
    # def get_state_id(self, action:Hashable)->int:
    #     if action not in self._state2id:
    #         self._action2id[action] = len(self._action2id)
    #         self._id2action[self._action2id[action]] = action
    #     return self._state2id[action]

    # def get_action(self, id:int)->Hashable:
    #     return self._id2action.get(id, None)

    def update_edge_reward(self, edge_id:tuple, reward)->None:
        if edge_id not in self._edge_reward:
            self._edge_reward
        else:
            self._edge_reward[edge_id] += (reward - self._edge_reward[edge_id])/self._edge_count[edge_id]
    
    def add_trajectory(self, trajectory:Iterable)->None:
        trajectory_return = 0
        self._starting_states.add(trajectory[0].state)
        for transition in trajectory:
            src_id = transition.state
            action_id = transition.action
            dst_id = transition.next_state
            reward = transition.reward
            # add the edge to the dicts
            if (src_id, dst_id, action_id) not in self._edge_count:
                self._edge_count[(src_id, dst_id, action_id)] = 0
            self._edge_count[(src_id, dst_id, action_id)] += 1
            # update the mean reward on the edge
            self.update_edge_reward((src_id, dst_id, action_id), reward)
            trajectory_return += reward
        self._terminal_states.add(trajectory[-1].next_state)
        self._returns.append(trajectory_return)
        self._lengths.append(len(trajectory))

    def get_probability_per_edge(self):
        """
        Computes the empirical probability of taking each adge from the src_node.
        """
        total_out_edges_use = {}
        for (src, _, _), frequency in self._edge_count.items():
            if src not in total_out_edges_use:
                total_out_edges_use[src] = 0
            total_out_edges_use[src] += frequency
        edge_probs = {}
        for (src,dst,action), value in self._edge_count.items():
            edge_probs[(src,dst,action)] = value/total_out_edges_use[src]
        return edge_probs

    @property
    def edge_set(self)->Iterable:
        """
        Produces set of edges in the transition graph
        """
        return set(self._edge_count.keys())
    
    @property
    def node_set(self)->Iterable:
        """
        Produces set of nodes in the transition graph
        """
        nodes = set()
        for (s,_,d) in self._edge_count.keys():
            nodes.add(s)
            nodes.add(d)
        return nodes
    
    @property
    def loops(self)->Iterable:
        """
        Produces set of loops in the transition graph
        """
        loops = set()
        for (s,a,d) in self._edge_count.keys():
            if s == d:
                loops.add((s,a,d))
        return loops

    @property
    def starting_states(self):
        return self._starting_states
    @property
    def terminal_states(self):
        return self._terminal_states

    def get_graph_metrics(self)->dict:
        metrics = {
            "nodes": len(self.node_set),
            "edges": len(self.edge_set),
            "loops": len(self.loops),
            "mean_winrate": np.mean(self._returns),
            "mean_trajectory_length": np.mean(self._lengths)
        }
        return metrics

    def get_added_nodes(self, other:object):
        """
        Finds nodes that this graph has in comparison to the 'other' graph.
        """
        if other and isinstance(other, TrajectoryGraph):
            return self.node_set - other.node_set
        else:
            raise ValueError("other must be class TrajectoryGraph")
    
    def get_removed_nodes(self, other:object):
        """
        Finds nodes that 'other' has in comparison to the this graph.
        """
        if other and isinstance(other, TrajectoryGraph):
            return other.node_set - self.node_set
        else:
            raise ValueError("other must be class TrajectoryGraph")

    def get_added_edges(self, other:object):
        """
        Finds edges that this graph has in comparison to the 'other' graph.
        """
        if other and isinstance(other, TrajectoryGraph):
            return self.edge_set - other.edge_set
        else:
            raise ValueError("other must be class TrajectoryGraph")
    
    def get_removed_edges(self, other:object):
        """
        Finds edges that 'other' has in comparison to the this graph.
        """
        if other and isinstance(other, TrajectoryGraph):
            return other.edge_set - self.edge_set
        else:
            raise ValueError("other must be class TrajectoryGraph")

    def compare_with_previous(self, other)->dict:
        diff = {
            "added_nodes": len(self.get_added_nodes(other)),
            "removed_nodes": len(self.get_removed_nodes(other)),
            "added_edges": len(self.get_added_edges(other)),
            "removed_edges":len(self.get_removed_edges(other)),
            "overlap_nodes":overlap_coefficient(self.node_set, other.node_set),
            "overlap_edges":overlap_coefficient(self.edge_set, other.edge_set),
            "jaccard_nodes": jaccard_similarity_coefficient(self.node_set, other.node_set),
            "jaccard_edges":jaccard_similarity_coefficient(self.edge_set, other.edge_set),
            "mean_crossentropy": graph_trainsion_cross_entropy(self, other),
            "winrate_diff": np.mean(self._returns)-np.mean(other._returns)
        }
        print(diff)
        return diff

# # # class CostMahalanobis(BaseCost):
# #     """Custom cost function using Mahalanobis distance."""

# #     model = "mahalanobis"
# #     min_size = 2

# #     def fit(self, signal):
# #         """Set the internal parameters and compute inverse covariance matrix."""
# #         self.signal = np.asarray(signal)
# #         self.cov_inv = np.linalg.inv(np.cov(self.signal.T))  # Compute inverse covariance
# #         return self

# #     def error(self, start, end):
# #         """Compute the sum of Mahalanobis distances for segment [start:end].

# #         Args:
# #             start (int): start of the segment
# #             end (int): end of the segment

# #         Returns:
# #             float: segment cost (sum of distances)
# #         """
# #         sub = self.signal[start:end]
# #         mean_vector = sub.mean(axis=0)  # Compute mean of the segment
# #         return sum(mahalanobis(x, mean_vector, self.cov_inv) for x in sub)

# def detect_stopping_cp(change_points, threshold=3):
#     all_cp = []
#     for cp_list in change_points.values():
#         all_cp += cp_list
#     all_cp = list(sorted(set(all_cp)))
#     stable_regions_size = [j-i for i,j in zip(all_cp, all_cp[1:])]
#     print(stable_regions_size, max(stable_regions_size), threshold)
#     stopping_checkpoint = None
#     if max(stable_regions_size) >= threshold:
#         for i,cp in enumerate(all_cp):
#             if stable_regions_size[i] >= threshold:
#                 stopping_checkpoint = cp+threshold
#                 print("Found stopping point:", stopping_checkpoint)
#                 break
#     return stopping_checkpoint

# def compute_mahalanobis_distance(signal):
#     n, d = signal.shape  # n: number of time steps, d: number of variables
#     distances = np.zeros((n - 1,))  # Store distances between consecutive points

#     # Compute covariance matrix and its inverse
#     cov_matrix = np.cov(signal.T)
#     cov_inv = np.linalg.inv(cov_matrix)
    
#     # Compute Mahalanobis distance for each pair of consecutive points
#     for i in range(n - 1):
#         dist = mahalanobis(signal[i], signal[i+1], cov_inv)
#         distances[i] = dist

#     return distances


# def compute_dynamic_penalty(signal, C=1):
#     """
#     Compute penalty dynamically based on the sequence length.
    
#     Parameters:
#     - signal: np.array, shape (T, d), where T is the number of time points and d is the number of features.
#     - C: tuning constant for the penalty (default is 3).
    
#     Returns:
#     - computed penalty value
#     """
#     # Assuming 'signal' is a 2D numpy array where each row is a time step and columns are the different dimensions
#     T, D = signal.shape  # T is the number of time steps, D is the number of dimensions

#     # # 1. Compute the standard deviation across each dimension
#     # std_dev = np.std(signal, axis=0)  # Standard deviation per dimension

#     # # 2. Compute the L2 norm of the standard deviations across all dimensions (multivariate std dev)
#     # std_dev_multivariate = np.sqrt(np.sum(std_dev**2))  # L2 norm of the std dev

#     # # 3. Define the max sequence length to scale penalty appropriately (example: set T_max = 100)
#     # T_max = 50  # You can define T_max based on your dataset or use the maximum length of the sequences you're working with

#     # # 4. Adjust penalty based on sequence length (using normalized or logarithmic scaling)
#     # penalty = C * std_dev_multivariate * np.sqrt(T / T_max)  # Scaling by max length
#     T = len(signal)  # Length of the sequence
#     sigma = np.std(signal, axis=0).mean()  # Mean standard deviation across features
#     C = 1  # Scaling factor (tune experimentally)

#     # Compute dynamic penalty
#     penalty = C *np.log(T)
#     return penalty

# def get_change_points(data:dict, model="l1", max_penalty=0.5):
#     # metrics = []
#     # for cp,cp_metric in data.items():
#     #     metrics.append([v for v in cp_metric.values()])
#     # metrics = np.array(metrics)
   
#     # mahalanobis_distances = compute_mahalanobis_distance(metrics)
#     # # Run PELT with the custom Mahalanobis cost function
#     # penalty = compute_dynamic_penalty(metrics,C=max_penalty)
#     # print("Computed penalty:", penalty)
#     # # algo = rpt.Pelt(model="l2", min_size=1, jump=1).fit(mahalanobis_distances)  # Use 'l2' model with precomputed distances
#     # algo = rpt.Pelt(custom_cost=CostMahalanobis(), jump=1).fit(metrics)
#     # change_points = algo.predict(pen=penalty)  # Adjust penalty for more/less detections
#     # print(change_points)
#     # return change_points
    
#     PELT_changepoints = {}
#     for metric_key in list(next(iter(data.values())).keys()):
#         metric_values = [data[cp][metric_key] for cp in range(len(data.keys()))]
#         if metric_key in ["wasserstein", "GED"]:
#             PELT_changepoints[metric_key] = PELT_changepoint(np.array(metric_values), max_penalty=2, mode="")
#         else:
#             PELT_changepoints[metric_key] = PELT_changepoint(np.array(metric_values), max_penalty=2, mode="")
#         print(PELT_changepoints[metric_key])
#     print(PELT_changepoints)
#     return PELT_changepoints

# def PELT_changepoint(data, model="l1", max_penalty=10, mode=""):
#     algo = rpt.Pelt(model=model, min_size=1, jump=1).fit(data)
#     sigma2 = np.var(data)
#     penalty = min(sigma2 * np.log(len(data)), max_penalty)
#     print("Computed penalty:", penalty)
#     change_points = algo.predict(pen=penalty)
#     filtered_change_points = []
    
#     for cp in change_points:
#         if cp > 0 and cp < len(data):# Ensure valid indices
#             if mode == "upward":
#                 if data[cp] > data[cp - 1]:  # Only keep rising changes
#                     filtered_change_points.append(cp)
#             elif mode == "downward":
#                 if data[cp] < data[cp - 1]:  # Only keep dropping changes
#                     filtered_change_points.append(cp)
#             else:
#                 filtered_change_points.append(cp)
#     return filtered_change_points

# def stable_regions_KernelCPD(data, kernel="linear", max_penalty=10):
#     # algo = rpt.KernelCPD(kernel="linear").fit(data)
#     # sigma2 = np.var(data)
#     # penalty = min(2*sigma2 * np.log(len(data)), max_penalty)
#     # print("Computed penalty:", penalty)
#     # change_points = algo.predict(pen=3)
#     algo = rpt.Pelt(model="l1", min_size=1, jump=1).fit(data)
#     sigma2 = np.var(data)
#     penalty = min(2*sigma2 * np.log(len(data)), max_penalty)
#     print("Computed penalty:", penalty)
#     change_points = algo.predict(pen=penalty)
#     return change_points

# class TrajectoryGraph:
#     def __init__(self)->None:
#         self._checkpoint_size = {}
#         self._checkpoint_edges = {}
#         self._checkpoint_nodes = {}
#         self._checkpoint_cumulative_rewards = {}
#         self._wins_per_checkpoint = {}
#         self._steps_per_checkpoint = {}
#         self._state_to_id = {}
#         self._id_to_state = {}
#         self._action_to_id = {}
#         self._id_to_action = {}
#         self._current_checkpoint = 0


#     def save_to_pickle(self, file_path):
#         """Save the object to a pickle file."""
#         with open(file_path, 'wb') as file:
#             pickle.dump(self, file)
#         print(f"Object saved to {file_path}")
    
#     @classmethod
#     def load_from_pickle(cls, file_path):
#         """Load an object from a pickle file."""
#         with open(file_path, 'rb') as file:
#             obj = pickle.load(file)
#         print(f"Object loaded from {file_path}")
#         return obj

#     @property
#     def num_checkpoints(self)->int:
#         return len(self._checkpoint_size.keys())

#     def get_state_id(self, state,state_encoder=None)->int:
#         """
#         Returns state id or creates new one if the state was not registered before
#         """
#         if state_encoder:
#             s = state_encoder(state)
#         else:
#             s = state
#         if s not in self._state_to_id.keys():
#             self._state_to_id[s] = len(self._state_to_id)
#             self._id_to_state[self._state_to_id[s]] = state
#         return self._state_to_id[s]
    
#     def get_state(self, id:int):
#         return self._id_to_state[id]

#     def get_action_id(self, action)->int:
#         """
#         Returns action id or creates new one if the state was not registered before
#         """
#         if action not in self._action_to_id.keys():
#             self._action_to_id[action] = len(self._action_to_id)
#             self._id_to_action[self._action_to_id[action]] = action
#         return self._action_to_id[action]

#     def get_action(self, id:int):
#         return self._id_to_action[id]

#     def move_to_next_checkpoint(self)->int:
#         self._current_checkpoint += 1
#         return self._current_checkpoint

#     def add_trajectory(self, transitions:list, checkpoint_id:int=None)->None:
#         """
#         Adds a trajectory to the graph
#         """
#         if checkpoint_id is None:
#             checkpoint_id = self._current_checkpoint
#         if checkpoint_id not in self._checkpoint_edges:
#             self._checkpoint_edges[checkpoint_id] = {}
#         if checkpoint_id not in self._checkpoint_nodes:
#             self._checkpoint_nodes[checkpoint_id] = {}
#         rewards = []
#         for src, action, reward, dst in transitions:
#             src_id = self.get_state_id(src)
#             action_id = self.get_action_id(action)
#             dst_id = self.get_state_id(dst)
#             rewards.append(reward)
#             if (src_id, dst_id, action_id) not in self._checkpoint_edges[checkpoint_id]:
#                 self._checkpoint_edges[checkpoint_id][src_id, dst_id, action_id] = 0
#             self._checkpoint_edges[checkpoint_id][src_id, dst_id, action_id] += 1
#             # add src_id and dst_id to nodes
#             if src_id not in self._checkpoint_nodes[checkpoint_id]:
#                 self._checkpoint_nodes[checkpoint_id][src_id] = 0
#             self._checkpoint_nodes[checkpoint_id][src_id] += 1
#             if dst_id not in self._checkpoint_nodes[checkpoint_id]:
#                 self._checkpoint_nodes[checkpoint_id][dst_id] = 0
#             self._checkpoint_nodes[checkpoint_id][dst_id] += 1
#         # increment the size of the checkpoint
#         if checkpoint_id not in self._checkpoint_size:
#             self._checkpoint_size[checkpoint_id] = 0
#         self._checkpoint_size[checkpoint_id] += 1
        
#         # add rewards to the checkpoint
#         if checkpoint_id not in self._checkpoint_cumulative_rewards:
#             self._checkpoint_cumulative_rewards[checkpoint_id] = []
#         self._checkpoint_cumulative_rewards[checkpoint_id].append(sum(rewards))
        
#         # add win/loss to the checkpoint
#         if checkpoint_id not in self._wins_per_checkpoint:
#             self._wins_per_checkpoint[checkpoint_id] = []
#         if rewards[-1] > 0:
#             self._wins_per_checkpoint[checkpoint_id].append(1)
#         else:
#             self._wins_per_checkpoint[checkpoint_id].append(0)
#         # add steps to the checkpoint
#         if checkpoint_id not in self._steps_per_checkpoint:
#             self._steps_per_checkpoint[checkpoint_id] = []
#         self._steps_per_checkpoint[checkpoint_id].append(len(rewards))

#     def add_checkpoint(self, trajectories:list, end_reason=None)->None:
#         print("adding checkpoint:", self.num_checkpoints)
#         checkpoint_id = self.num_checkpoints
#         for trajectory in trajectories:
#             # ignore trajectories that did not finish with selected end_reason
#             if len(trajectory) == 0:
#                 continue
#             if end_reason and trajectory["end_reason"] not in end_reason:
#                 continue
#             self.add_trajectory(trajectory, checkpoint_id)

#     def build_simplified_graph(self, checkpoint_id:int)->nx.DiGraph:
#         """
#         Builds a directed graph for a given checkpoint
#         """
#         G = nx.DiGraph()
#         for (s1,s2), freq in self.get_checkpoint_simple_edges(checkpoint_id).items():
#             if G.has_edge(s1, s2):
#                 G[s1][s2]['weight'] += freq  # Increment weight if transition already exists
#             else:
#                 G.add_edge(s1, s2, weight=freq)  # Add new transition
#         return G

#     def build_graph(self, checkpoint_id:int)->nx.DiGraph:
#         """
#         Builds a directed graph for a given checkpoint
#         """
#         G = nx.DiGraph()
#         for (s1,s2,a), freq in self.get_outgoing_edge_probs(checkpoint_id).items():
#             if G.has_edge(s1, s2):
#                 G[s1][s2]['weight'] += freq
#             else:
#                 G.add_edge(s1, s2, weight=freq, action=a)
#         return G
    
#     def plot_graph(self, checkpoint_id:int, fig_sufix="")->None:
#         """
#         Plots a graph for a given checkpoint
#         """
#         G = self.build_graph(checkpoint_id)

#         plt.figure(figsize=(32, 24))
#         pos = nx.planar_layout(G)
#         nx.draw(G, pos, with_labels=True, node_size=200, node_color="lightblue", edge_color="gray", arrowsize=15)
#         edge_labels = {(s1, s2): f"{G[s1][s2]['action']} ({G[s1][s2]['weight']})" for s1, s2 in G.edges}
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
#         plt.title("Transition Graph of RL Agent")
#         plt.savefig(f"figures/transition_graph_checkpoint_{checkpoint_id}{'_' + fig_sufix if len(fig_sufix) > 0 else ''}.png")

#     def plot_simplified_graph(self, checkpoint_id:int, fig_sufix="")->None:
#         """
#         Plots a simplified graph for a given checkpoint
#         """
#         G = self.build_simplified_graph(checkpoint_id)

#         plt.figure(figsize=(32, 24))
#         pos = nx.planar_layout(G)  # Graph layout
#         nx.draw(G, pos, with_labels=True, node_size=200, node_color="lightblue", edge_color="gray", arrowsize=15)
#         edge_labels = {(s1, s2): f"{G[s1][s2]['weight']}" for s1, s2 in G.edges}
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
#         plt.title("Transition Graph of RL Agent")
#         plt.savefig(f"figures/simplified_transition_graph_checkpoint_{checkpoint_id}{'_' + fig_sufix if len(fig_sufix) > 0 else ''}.png")

#     def get_checkpoint_wr(self, checkpoint_id:int)->tuple:
#         """
#         Calculate the mean and standard deviation of wins for a given checkpoint.

#         Args:
#             checkpoint_id (int): The ID of the checkpoint to retrieve statistics for.

#         Returns:
#             tuple: A tuple containing the mean and standard deviation of wins for the specified checkpoint.

#         Raises:
#             IndexError: If the checkpoint_id is not found in the wins per checkpoint data.
#         """
#         if checkpoint_id not in self._wins_per_checkpoint:
#             raise IndexError(f"Checkpoint id '{checkpoint_id}' not found!")
#         else:
#             return np.mean(self._wins_per_checkpoint[checkpoint_id]), np.std(self._wins_per_checkpoint[checkpoint_id])

#     def get_checkpoint_rewards(self, checkpoint_id:int)->tuple:
#         """
#         Calculate the mean and standard deviation of rewards for a given checkpoint.
#         Args:
#             checkpoint_id (int): The ID of the checkpoint to retrieve rewards for.
#         Returns:
#             tuple: A tuple containing the mean and standard deviation of the rewards.
#         Raises:
#             IndexError: If the checkpoint_id is not found in the cumulative rewards.
#         """
#         if checkpoint_id not in self._checkpoint_cumulative_rewards:
#             raise IndexError(f"Checkpoint id '{checkpoint_id}' not found!")
#         else:
#             return np.mean(self._checkpoint_cumulative_rewards[checkpoint_id]), np.std(self._checkpoint_cumulative_rewards[checkpoint_id])

#     def get_checkpoint_steps(self, checkpoint_id:int)->tuple:
#         """
#         Calculate the mean and standard deviation of steps for a given checkpoint.
#         Args:
#             checkpoint_id (int): The ID of the checkpoint to retrieve steps for.
#         Returns:
#             tuple: A tuple containing the mean and standard deviation of the steps.
#         Raises:
#             IndexError: If the checkpoint_id is not found in the steps per checkpoint data.
#         """
#         if checkpoint_id not in self._steps_per_checkpoint:
#             raise IndexError(f"Checkpoint id '{checkpoint_id}' not found!")
#         else:
#             return np.mean(self._steps_per_checkpoint[checkpoint_id]), np.std(self._steps_per_checkpoint[checkpoint_id])
    
#     def get_checkpoint_edges(self, checkpoint_id:int)->dict:
#         """
#         Returns the edges of a checkpoint
#         """
#         return self._checkpoint_edges[checkpoint_id]

#     def get_checkpoint_nodes(self, checkpoint_id:int)->dict:
#         """
#         Returns the nodes of a checkpoint
#         """
#         return self._checkpoint_nodes[checkpoint_id]

#     def get_checkpoint_simple_edges(self, checkpoint_id:int)->dict:
#         """
#         Returns the simple edges of a checkpoint
#         """
#         simple_edges = {}
#         for (src, dst, _), occurence in self._checkpoint_edges[checkpoint_id].items():
#             simple_edges[src, dst] = simple_edges.get((src, dst), 0) + occurence
#         return simple_edges
    
#     def get_checkpoint_loops(self, checkpoint_id:int)->dict:
#         """
#         Returns the loops of a checkpoint
#         """
#         loops = {}
#         for (src, dst, action), occurence in self._checkpoint_edges[checkpoint_id].items():
#             if src == dst:
#                 loops[src, action, dst] = occurence
#         return loops

#     def get_checkpoint_simple_loops(self, checkpoint_id:int)->dict:
#         """
#         Returns the simple loops of a checkpoint
#         """
#         simple_loops = {}
#         for (src,dst,_), occurence in self.get_checkpoint_loops(checkpoint_id).items():
#             simple_loops[src, dst] = simple_loops.get((src, dst), 0) + occurence
#         return simple_loops
    
#     def get_graph_stats_progress(self):
#         ret = {}
#         print("Checkpoint,\tWR,\tEdges,\tSimpleEdges,\tNodes,\tLoops,\tSimpleLoops")
#         for i in range(self.num_checkpoints):
#             data = self.get_checkpoint_stats(i)
#             ret[i] = data
#             print(",".join(f"{k}:{v:.2f}" for k,v in data.items()))
#         return ret

#         data = self.get_graph_stats_progress()
#         wr = [data[i]["winrate"] for i in range(len(data))]
#         num_nodes = [data[i]["num_nodes"] for i in range(len(data))]
#         num_edges = [data[i]["num_edges"] for i in range(len(data))]
#         num_simle_edges = [data[i]["num_simplified_edges"] for i in range(len(data))]
#         num_loops = [data[i]["num_loops"] for i in range(len(data))]
#         num_simplified_loops  =  [data[i]["num_simplified_loops"] for i in range(len(data))]
#         steps = [data[i]["steps"] for i in range(len(data))]
#         checkpoints = range(self.num_checkpoints) 
#         plt.plot(checkpoints, num_nodes, label='Number of nodes')
#         plt.plot(checkpoints, num_edges, label='Number of edges')
#         plt.plot(checkpoints, num_simle_edges, label='Number of simplified edges')
#         plt.plot(checkpoints, num_loops, label='Number of loops')
#         plt.plot(checkpoints, num_simplified_loops, label='Number of simplified loops')
#         plt.plot(checkpoints, steps, label='Number of steps')

#         plt.title("Graph statistics per checkpoint")
#         plt.yscale('log')
#         plt.xlabel("Checkpoints")
#         # Show legend
#         plt.legend()

#         # Save the figure as an image file
#         plt.savefig(os.path.join(filedir, f"{filename}{'_'+file_suffix if file_suffix else ''}.png"))

#     def get_checkpoint_stats(self, checkpoint_id:int)->dict:
#         if checkpoint_id not in self._wins_per_checkpoint:
#             raise IndexError(f"Checkpoint id '{checkpoint_id}' not found!")
#         else:
#             data = {}
#             wr, wr_std = self.get_checkpoint_wr(checkpoint_id)
#             data["winrate"] = wr*100
#             data["winrate_std"] = wr_std*100
#             data["num_edges"] = len(self.get_checkpoint_edges(checkpoint_id))
#             data["num_simplified_edges"] = len(self.get_checkpoint_simple_edges(checkpoint_id))
#             data["num_nodes"] = len(self.get_checkpoint_nodes(checkpoint_id))
#             data["num_loops"] = len(self.get_checkpoint_loops(checkpoint_id))
#             data["num_simplified_loops"] = len(self.get_checkpoint_simple_loops(checkpoint_id))
#             rewards, rewards_std = self.get_checkpoint_rewards(checkpoint_id)
#             data["rewards"] = rewards
#             data["rewards_std"] = rewards_std
#             steps, steps_std = self.get_checkpoint_steps(checkpoint_id)
#             data["steps"] = steps
#             data["steps_std"] = steps_std
#             return data

#     def get_tg_stats_per_checkpoint(self):
#         stats = {}
#         for i in range(0, self.num_checkpoints):
#             stats[i] = self.get_checkpoint_stats(i)
#         return stats

#     def plot_tg_stats_per_checkpoint(self, fig_sufix=""):
#         stats = self.get_tg_stats_per_checkpoint()
        
#         checkpoints = list(stats.keys())
#         stat_keys = list(next(iter(stats.values())).keys())

#         fig, ax = plt.subplots(figsize=(14, 8))
#         for stat_key in stat_keys:
#             stat_values = [stats[cp][stat_key] for cp in checkpoints]
#             if ("std" in stat_key):
#                 continue
#             match stat_key:
#                 case "winrate" | "winrate_std":
#                     marker = 'o'
#                     linestyle = ':'
#                 case "rewards" | "rewards_std":
#                     marker = 'o'
#                     linestyle = '-.'
#                 case "steps" | "steps_std":
#                     marker = 'o'
#                     linestyle = '--'
#                 case _:
#                     marker = 'o'
#                     linestyle = '-'
#             ax.plot(checkpoints, stat_values, label=stat_key, marker=marker, linestyle=linestyle)
#         ax.set_xlabel('Checkpoint ID')
#         ax.set_ylabel('Value')
#         ax.set_title('TG Stats per Checkpoint')
#         ax.legend()
#         ax.grid(True)
#         plt.tight_layout()
#         plt.savefig(f"figures/tg_stats_per_checkpoint_{fig_sufix}.png")
    
#     def get_change_in_nodes(self, checkpoint1, checkpoint2):
#         nodes1 = self.get_checkpoint_nodes(checkpoint1).keys()
#         nodes2 = self.get_checkpoint_nodes(checkpoint2).keys()
#         added_nodes = set(nodes2) - set(nodes1)
#         removed_nodes = set(nodes1) - set(nodes2)
#         return added_nodes, removed_nodes

#     def get_change_in_edges(self, checkpoint1, checkpoint2):
#         edges1 = self.get_checkpoint_edges(checkpoint1).keys()
#         edges2 = self.get_checkpoint_edges(checkpoint2).keys()
#         added_edges = set(edges2) - set(edges1)
#         removed_edges = set(edges1) - set(edges2)
#         return added_edges, removed_edges

#     def get_change_in_simple_edges(self, checkpoint1, checkpoint2):
#         edges1 = self.get_checkpoint_simple_edges(checkpoint1)
#         edges2 = self.get_checkpoint_simple_edges(checkpoint2)
#         added_simple_edges = set(edges2.keys()) - set(edges1.keys())
#         removed_simple_edges = set(edges1.keys()) - set(edges2.keys())
#         return added_simple_edges, removed_simple_edges

#     def get_change_in_loops(self, checkpoint1, checkpoint2):
#         loops = self.get_checkpoint_loops(checkpoint1).keys()
#         loops2 = self.get_checkpoint_loops(checkpoint2).keys()
#         added_loops = set(loops2) - set(loops)
#         removed_loops = set(loops) - set(loops2)
#         return added_loops, removed_loops

#     def get_change_in_simple_loops(self, checkpoint1, checkpoint2):
#         loops = self.get_checkpoint_simple_loops(checkpoint1)
#         loops2 = self.get_checkpoint_simple_loops(checkpoint2)
#         added_simple_loops = set(loops2.keys()) - set(loops.keys())
#         removed_simple_loops = set(loops.keys()) - set(loops2.keys())
#         return added_simple_loops, removed_simple_loops
    
#     def get_changes_per_checkpoint(self):
#         changes = {}
#         for i in range(1, self.num_checkpoints):
#             added_nodes, removed_nodes = self.get_change_in_nodes(i-1, i)
#             added_edges, removed_edges = self.get_change_in_edges(i-1, i)
#             added_loops, removed_loops = self.get_change_in_loops(i-1, i)
#             added_simple_edges, removed_simple_edges = self.get_change_in_simple_edges(i-1, i)
#             added_simple_loops, removed_simple_loops = self.get_change_in_simple_loops(i-1, i)
#             winrate = np.mean(self._wins_per_checkpoint[i]) - np.mean(self._wins_per_checkpoint[i-1])
#             steps = np.mean(self._steps_per_checkpoint[i]) - np.mean(self._steps_per_checkpoint[i-1])
#             changes[i] = {
#                 "added_nodes": added_nodes,
#                 "removed_nodes": removed_nodes,
#                 "added_edges": added_edges,
#                 "removed_edges": removed_edges,
#                 "added_self_loops": added_loops,
#                 "removed_self_loops": removed_loops,
#                 "added_simple_edges": added_simple_edges,
#                 "removed_simple_edges": removed_simple_edges,
#                 "added_simple_loops": added_simple_loops,
#                 "removed_simple_loops": removed_simple_loops,
#                 "winrate_change": winrate,
#                 "steps_change": steps
#             }
#         return changes

#     def plot_changes_per_checkpoint(self, fig_sufix=""):
#         changes = self.get_changes_per_checkpoint()
        
#         checkpoints = list(changes.keys())
#         added_nodes = [len(changes[cp]['added_nodes']) for cp in checkpoints]
#         removed_nodes = [len(changes[cp]['removed_nodes']) for cp in checkpoints]
#         added_edges = [len(changes[cp]['added_edges']) for cp in checkpoints]
#         removed_edges = [len(changes[cp]['removed_edges']) for cp in checkpoints]
#         added_self_loops = [len(changes[cp]['added_self_loops']) for cp in checkpoints]
#         removed_self_loops = [len(changes[cp]['removed_self_loops']) for cp in checkpoints]
#         added_simple_edges = [len(changes[cp]['added_simple_edges']) for cp in checkpoints]
#         removed_simple_edges = [len(changes[cp]['removed_simple_edges']) for cp in checkpoints]

#         plt.figure(figsize=(12, 8))

#         plt.plot(checkpoints, added_nodes, label='Added Nodes', marker='o')
#         plt.plot(checkpoints, removed_nodes, label='Removed Nodes', marker='D')
#         plt.plot(checkpoints, added_edges, label='Added Edges', marker='o')
#         plt.plot(checkpoints, removed_edges, label='Removed Edges', marker='D')
#         plt.plot(checkpoints, added_self_loops, label='Added Self-Loops', marker='o')
#         plt.plot(checkpoints, removed_self_loops, label='Removed Self-Loops', marker='D')
#         plt.plot(checkpoints, added_simple_edges, label='Added Simple Edges', marker='o')
#         plt.plot(checkpoints, removed_simple_edges, label='Removed Simple Edges', marker='D')

#         plt.xlabel('Checkpoint')
#         plt.ylabel('Count')
#         plt.title('Changes in Nodes, Edges, and Self-Loops per Checkpoint')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"figures/changes_per_checkpoint_{fig_sufix}.png")

#     def plot_changes_per_checkpoint_barplot(self):
#         changes = self.get_changes_per_checkpoint()
        
#         checkpoints = list(changes.keys())
#         added_nodes = [len(changes[cp]['added_nodes']) for cp in checkpoints]
#         removed_nodes = [len(changes[cp]['removed_nodes']) for cp in checkpoints]
#         added_edges = [len(changes[cp]['added_edges']) for cp in checkpoints]
#         removed_edges = [len(changes[cp]['removed_edges']) for cp in checkpoints]
#         added_self_loops = [len(changes[cp]['added_self_loops']) for cp in checkpoints]
#         removed_self_loops = [len(changes[cp]['removed_self_loops']) for cp in checkpoints]
#         added_simple_edges = [len(changes[cp]['added_simple_edges']) for cp in checkpoints]
#         removed_simple_edges = [len(changes[cp]['removed_simple_edges']) for cp in checkpoints]

#         bar_width = 0.1
#         index = np.arange(len(checkpoints))

#         fig, ax = plt.subplots(figsize=(14, 8))

#         bar1 = ax.bar(index - 2*bar_width, added_nodes, bar_width, label='Added Nodes')
#         bar2 = ax.bar(index - bar_width, -np.array(removed_nodes), bar_width, label='Removed Nodes')
#         bar3 = ax.bar(index, added_edges, bar_width, label='Added Edges')
#         bar4 = ax.bar(index + bar_width, -np.array(removed_edges), bar_width, label='Removed Edges')
#         bar5 = ax.bar(index + 2*bar_width, added_self_loops, bar_width, label='Added Self-Loops')
#         bar6 = ax.bar(index + 3*bar_width, -np.array(removed_self_loops), bar_width, label='Removed Self-Loops')
#         bar7 = ax.bar(index + 4*bar_width, added_simple_edges, bar_width, label='Added Simple Edges')
#         bar8 = ax.bar(index + 5*bar_width, -np.array(removed_simple_edges), bar_width, label='Removed Simple Edges')

#         ax.set_xlabel('Checkpoint')
#         ax.set_ylabel('Count')
#         ax.set_title('Changes in Nodes, Edges, and Self-Loops per Checkpoint')
#         ax.set_xticks(index)
#         ax.set_xticklabels(checkpoints)
#         ax.legend()
#         ax.grid(True)
#         plt.savefig("figures/changes_per_checkpoint_bar.png")

#     def plot_changes_per_checkpoint_combined(self, fig_sufix=""):
#         changes = self.get_changes_per_checkpoint()
        
#         checkpoints = list(changes.keys())
#         added_nodes = [len(changes[cp]['added_nodes']) for cp in checkpoints]
#         removed_nodes = [len(changes[cp]['removed_nodes']) for cp in checkpoints]
#         added_edges = [len(changes[cp]['added_edges']) for cp in checkpoints]
#         removed_edges = [len(changes[cp]['removed_edges']) for cp in checkpoints]
#         added_self_loops = [len(changes[cp]['added_self_loops']) for cp in checkpoints]
#         removed_self_loops = [len(changes[cp]['removed_self_loops']) for cp in checkpoints]
#         added_simple_edges = [len(changes[cp]['added_simple_edges']) for cp in checkpoints]
#         removed_simple_edges = [len(changes[cp]['removed_simple_edges']) for cp in checkpoints]

#         bar_width = 0.1
#         index = np.arange(len(checkpoints))

#         fig, ax1 = plt.subplots(1, 1, figsize=(14, 16))

#         # Line plot
#         ax1.plot(checkpoints, added_nodes, label='Added Nodes', marker='o')
#         ax1.plot(checkpoints, removed_nodes, label='Removed Nodes', marker='D')
#         ax1.plot(checkpoints, added_edges, label='Added Edges', marker='o')
#         ax1.plot(checkpoints, removed_edges, label='Removed Edges', marker='D')
#         ax1.plot(checkpoints, added_self_loops, label='Added Self-Loops', marker='o')
#         ax1.plot(checkpoints, removed_self_loops, label='Removed Self-Loops', marker='D')
#         ax1.plot(checkpoints, added_simple_edges, label='Added Simple Edges', marker='o')
#         ax1.plot(checkpoints, removed_simple_edges, label='Removed Simple Edges', marker='D')

#         ax1.set_xlabel('Checkpoint')
#         ax1.set_xticks(np.arange(0, len(checkpoints)+1, step=1))
#         ax1.set_ylabel('Count')
#         ax1.set_title('Changes in Nodes, Edges, and Self-Loops per Checkpoint (Line Plot)')
#         ax1.legend()
#         ax1.grid(True)

#         # # Bar plot
#         # bar1 = ax2.bar(index - 2*bar_width, added_nodes, bar_width, label='Added Nodes')
#         # bar2 = ax2.bar(index - bar_width, -np.array(removed_nodes), bar_width, label='Removed Nodes')
#         # bar3 = ax2.bar(index, added_edges, bar_width, label='Added Edges')
#         # bar4 = ax2.bar(index + bar_width, -np.array(removed_edges), bar_width, label='Removed Edges')
#         # bar5 = ax2.bar(index + 2*bar_width, added_self_loops, bar_width, label='Added Self-Loops')
#         # bar6 = ax2.bar(index + 3*bar_width, -np.array(removed_self_loops), bar_width, label='Removed Self-Loops')
#         # bar7 = ax2.bar(index + 4*bar_width, added_simple_edges, bar_width, label='Added Simple Edges')
#         # bar8 = ax2.bar(index + 5*bar_width, -np.array(removed_simple_edges), bar_width, label='Removed Simple Edges')

#         # ax2.set_xlabel('Checkpoint')
#         # ax2.set_ylabel('Count')
#         # ax2.set_title('Changes in Nodes, Edges, and Self-Loops per Checkpoint (Bar Plot)')
#         # ax2.set_xticks(index)
#         # ax2.set_xticklabels(checkpoints)
#         # ax2.legend()
#         # ax2.grid(True)

#         plt.tight_layout()
#         plt.savefig(f"figures/changes_per_checkpoint_combined_{fig_sufix}.png")
     
#     def get_outgoing_edge_probs(self, checkpoint_id:int)->dict:
#         """
#         Returns the probabilities of taking each edge from a source node in a checkpoint
#         """
#         edge_list = self.get_checkpoint_edges(checkpoint_id)
#         total_out_edges_use = {}
#         for (src, _, _), frequency in edge_list.items():
#             if src not in total_out_edges_use:
#                 total_out_edges_use[src] = 0
#             total_out_edges_use[src] += frequency
#         edge_probs = {}
#         for (src,dst,action), value in edge_list.items():
#             edge_probs[(src,dst,action)] = value/total_out_edges_use[src]
#         return edge_probs

#     def get_outgoing_simple_edge_probs(self, checkpoint_id:int)->dict:
#         """
#         Returns the probabilities of taking each simple edge from a source node in a checkpoint
#         """
#         simpl_edge_list = self.get_checkpoint_simple_edges(checkpoint_id)
#         total_out_edges_use = {}
#         for (src, _), frequency in simpl_edge_list.items():
#             if src not in total_out_edges_use:
#                 total_out_edges_use[src] = 0
#             total_out_edges_use[src] += frequency
#         simple_edge_probs = {}
#         for (src,dst), value in simpl_edge_list.items():
#             simple_edge_probs[(src,dst)] = value/total_out_edges_use[src]
#         return simple_edge_probs

#     def get_outgoing_edge_probs_progress(self)->dict:
#         """
#         Returns the probabilities of taking each edge from a source node in each checkpoint
#         """
#         all_edges = set().union(*(inner_dict.keys() for inner_dict in self._checkpoint_edges.values()))
#         edge_probs = {e: np.zeros(self.num_checkpoints, dtype=float) for e in all_edges}
#         for checkpoint_id in range(self.num_checkpoints):
#            cp_probs = self.get_outgoing_edge_probs(checkpoint_id)
#            for edge, prob in cp_probs.items():
#                 edge_probs[edge][checkpoint_id] = prob
#         return edge_probs

#     def get_outgoing_simple_edge_probs_progress(self)->dict:
#         """
#         Returns the probabilities of taking each simple edge from a source node in each checkpoint
#         """
#         all_edges = set().union(*(inner_dict.keys() for inner_dict in self._checkpoint_edges.values()))
#         simple_edges = set({(src,dst) for src, dst, _ in all_edges})
#         edge_probs = {e: np.zeros(self.num_checkpoints, dtype=float) for e in simple_edges}
#         for checkpoint_id in range(self.num_checkpoints):
#            cp_probs = self.get_outgoing_simple_edge_probs(checkpoint_id)
#            for edge, prob in cp_probs.items():
#                 edge_probs[edge][checkpoint_id] = prob
#         return edge_probs

#     def get_edge_frequency_in_checkpoint(self, checkpoint_id:int, upper_limit=1)->dict:
#         """
#         Returns the frequency of each edge in a checkpoint
#         """
#         edge_frequencies = {}
#         for (src, dst, action), count in self.get_checkpoint_edges(checkpoint_id).items():
#             edge_frequencies[(src, dst, action)] = min(count/self._checkpoint_size[checkpoint_id], upper_limit)
#         return edge_frequencies
    
#     def get_node_frequency_in_checkpoint(self, checkpoint_id:int, upper_limit=1)->dict:
#         """
#         Returns the frequency of each node in a checkpoint
#         """
#         node_frequencies = {}
#         for node, count in self.get_checkpoint_nodes(checkpoint_id).items():
#             node_frequencies[node] = min(count/self._checkpoint_size[checkpoint_id], upper_limit)
#         return node_frequencies
    
#     def get_edge_frequency_progress(self)->dict:
#         """
#         Returns the frequency of each edge in each checkpoint
#         """
#         all_edges = set().union(*(inner_dict.keys() for inner_dict in self._checkpoint_edges.values()))
#         edge_frequencies = {e: np.zeros(self.num_checkpoints, dtype=float) for e in all_edges}
#         for checkpoint_id in range(self.num_checkpoints):
#            cp_frequencies = self.get_edge_frequency_in_checkpoint(checkpoint_id)
#            for edge, freq in cp_frequencies.items():
#                 edge_frequencies[edge][checkpoint_id] = freq
#         return edge_frequencies

#     def get_node_frequency_progress(self)->dict:
#         """
#         Returns the frequency of each node in each checkpoint
#         """
#         all_nodes = set().union(*(inner_dict.keys() for inner_dict in self._checkpoint_nodes.values()))
#         node_frequencies = {e: np.zeros(self.num_checkpoints, dtype=float) for e in all_nodes}
#         for checkpoint_id in range(self.num_checkpoints):
#            cp_frequencies = self.get_node_frequency_in_checkpoint(checkpoint_id)
#            for node, freq in cp_frequencies.items():
#                 node_frequencies[node][checkpoint_id] = freq
#         return node_frequencies

 
#         node_probs = self.calculate_node_play_likelihoods()
#         edge_probs = self.calculate_edge_play_likelihoods()

#         checkpoints = self._checkpoints.keys()
#         nodes = [n for n in node_probs.keys() if np.count_nonzero(node_probs[n]) > self.num_checkpoints // 2]
#         edges = [e for e in edge_probs.keys() if np.count_nonzero(edge_probs[e]) > self.num_checkpoints // 2]

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

#         # Plot node probabilities
#         for node in nodes:
#             probs = np.clip(node_probs[node],None, 1)
#             ax1.plot(checkpoints,probs, label=f'Node {node}', marker='o')

#         ax1.set_xlabel('Checkpoint')
#         ax1.set_ylabel('Probability')
#         ax1.set_title('Node Play Likelihoods per Checkpoint')
#         ax1.legend()
#         ax1.grid(True)

#         # Plot edge probabilities
#         for edge in edges:
#             edge_prob_values = np.clip(edge_probs[edge],None, 1)
#             ax2.plot(checkpoints, edge_prob_values, label=f'Edge {edge}', marker='o')

#         ax2.set_xlabel('Checkpoint')
#         ax2.set_ylabel('Probability')
#         ax2.set_title('Edge Play Likelihoods per Checkpoint')
#         ax2.legend()
#         ax2.grid(True)

#         plt.tight_layout()
#         plt.savefig(f"figures/probability_changes_{fig_sufix}.png")

#     def plot_outgoing_edge_probs_heatmap(self, fig_sufix=""):
#         edge_probs = self.get_edge_frequency_progress()
#         #edge_probs_with_a_type = {(s,d,a,self._id_to_action(a).action_type): v for (s,d,a), v in edge_probs.items()}
#         fig, ax = plt.subplots(figsize=(14, 16))
#         probs = np.stack(list(edge_probs.values()), axis=0)
#         cax1 = ax.matshow(probs, cmap='viridis', aspect='auto')
#         fig.colorbar(cax1, ax=ax)
#         ax.set_xticks(np.arange(self.num_checkpoints))
#         ax.set_xticklabels(np.arange(self.num_checkpoints))
#         ax.set_yticks(np.arange(len(edge_probs.keys())))
#         ax.set_yticklabels(edge_probs.keys())
#         ax.set_xlabel('Checkpoints')
#         ax.set_ylabel('Edges')
#         ax.set_title('Outgoing Edge Probabilities per Checkpoint')
#         plt.tight_layout()  
#         plt.savefig(f"figures/outgoing_edge_probs{'' if len(fig_sufix)==0 else '_'+fig_sufix}.png")

    
#         node_probs = self.calculate_node_play_likelihoods()
#         edge_probs = self.calculate_edge_play_likelihoods()
#         checkpoints = self._checkpoints.keys()
#         nodes = [n for n in node_probs.keys()]
#         edges = [e for e in edge_probs.keys()]
#         # Plot EMA of node probabilities
#         data_nodes = np.array([node_probs[node] for node in nodes])
#         ema = self.compute_ema(data_nodes, span=2)
#         print(ema.shape)

#         fig, ax = plt.subplots(figsize=(14, 8))
#         for i, node in enumerate(node_probs.keys()):
#             ax.plot(checkpoints, ema[i,:], label=f'EMA {node}', marker='o')

#         ax.set_xlabel('Checkpoint')
#         ax.set_ylabel('EMA')
#         ax.set_title('Exponential Moving Average of Node Probabilities per Checkpoint')
#         ax.legend()
#         ax.grid(True)
#         plt.tight_layout()
#         plt.savefig("figures/ema_node_probabilities.png")
#         plt.show()       

#     def get_graph_metrics_per_checkpoint(self, state_dist_fn)->dict:
#         """
#         Computes graph metrics for each checkpoint and compares them
#         """
#         def directed_jaccard_similarity(G1, G2):
#             """Compute Jaccard similarity for directed graphs based on directed edge presence."""
#             edges1 = set(G1.edges)  # Directed edge tuples (s1, s2)
#             edges2 = set(G2.edges)

#             intersection = len(edges1 & edges2)  # Count overlapping directed edges
#             union = len(edges1 | edges2)  # Count total unique directed edges

#             return intersection / union if union > 0 else 0
        
        
#         def directed_preferential_attachment_between_graphs(G1, G2):
#             """
#             Compute the mean normalized directed preferential attachment score between two graphs.
            
#             For each edge (u,v) in the union of G1 and G2, the score is:
#                 PA(u, v) = d_out^{G1}(u) * d_in^{G2}(v)
#             and we normalize it by dividing by the maximum possible PA score in the graphs.
            
#             Returns:
#             --------
#             mean_normalized_pa : float
#                 The mean normalized PA score.
#             """
#             # Determine maximum out-degree in G1 and maximum in-degree in G2
#             max_out = max(dict(G1.out_degree()).values(), default=0)
#             max_in = max(dict(G2.in_degree()).values(), default=0)
            
#             # If either graph has no nodes, avoid division by zero.
#             if max_out == 0 or max_in == 0:
#                 return 0
            
#             PA_max = max_out * max_in
            
#             # Get the union of edge sets
#             edges_union = set(G1.edges()) | set(G2.edges())
#             pa_scores = []
            
#             for u, v in edges_union:
#                 out_deg = G1.out_degree(u) if G1.has_node(u) else 0
#                 in_deg = G2.in_degree(v) if G2.has_node(v) else 0
#                 pa_score = out_deg * in_deg
#                 normalized_pa = pa_score / PA_max
#                 pa_scores.append(normalized_pa)
            
#             return sum(pa_scores) / len(pa_scores) if pa_scores else 0
        
#         def graph_overlap_coefficient(G1, G2):
#             """
#             Compute Overlap Coefficient between two directed RL transition graphs.

#             :param G1: Directed NetworkX graph at checkpoint t
#             :param G2: Directed NetworkX graph at checkpoint t+1
#             :return: Overlap Coefficient (float)
#             """
#             edges_G1 = set(G1.edges())  # Get directed edges (state transitions)
#             edges_G2 = set(G2.edges())

#             intersection = len(edges_G1 & edges_G2)  # Shared edges
#             min_size = min(len(edges_G1), len(edges_G2))  # Normalize

#             return intersection / min_size if min_size > 0 else 0
#         def graph_overlap_coefficient_nodes(G1, G2):
#             """
#             Compute Overlap Coefficient between nodes of two directed RL transition graphs.

#             :param G1: Directed NetworkX graph at checkpoint t
#             :param G2: Directed NetworkX graph at checkpoint t+1
#             :return: Overlap Coefficient (float)
#             """
#             nodes_G1 = set(G1.nodes())  # Get directed edges (state transitions)
#             nodes_G2 = set(G2.nodes())

#             intersection = len(nodes_G1 & nodes_G2)  # Shared edges
#             min_size = min(len(nodes_G1), len(nodes_G2))  # Normalize

#             return intersection / min_size if min_size > 0 else 0
#         def graph_dice_coefficient(G1, G2):
#             """
#             Compute Dice Coefficient between two directed RL transition graphs.

#             :param G1: Directed NetworkX graph at checkpoint t
#             :param G2: Directed NetworkX graph at checkpoint t+1
#             :return: Dice Coefficient (float)
#             """
#             edges_G1 = set(G1.edges())  # Extract directed edges (state transitions)
#             edges_G2 = set(G2.edges())

#             intersection = len(edges_G1 & edges_G2)  # Shared edges
#             total_edges = len(edges_G1) + len(edges_G2)  # Total number of edges in both graphs

#             return (2 * intersection) / total_edges if total_edges > 0 else 0

#         def compare_katz_centrality(G1, G2):
#             """
#             Compute the Pearson correlation between Katz centrality scores of two directed graphs.
            
#             :param G1: Directed NetworkX graph at checkpoint t
#             :param G2: Directed NetworkX graph at checkpoint t+1
#             :return: Pearson correlation coefficient (-1 to 1)
#             """
#             # Compute Katz Centrality for both graphs
#             katz_G1 = nx.katz_centrality(G1, alpha=0.01, beta=1.0)
#             katz_G2 = nx.katz_centrality(G2, alpha=0.01, beta=1.0)
            
#             # Get the union of all nodes appearing in either graph
#             all_nodes = set(katz_G1.keys()).union(set(katz_G2.keys()))
            
#             # Create aligned lists of centrality scores (0 if node is missing in a graph)
#             scores_G1 = np.array([katz_G1.get(node, 0) for node in all_nodes])
#             scores_G2 = np.array([katz_G2.get(node, 0) for node in all_nodes])
            
#             # Compute Pearson correlation
#             correlation, _ = pearsonr(scores_G1, scores_G2)
            
#             return correlation

#         def compute_graph_edit_distance(G1, G2, timeout=5):
#             """
#             Compute Graph Edit Distance (GED) between two RL transition graphs.

#             :param G1: Directed NetworkX graph at checkpoint t
#             :param G2: Directed NetworkX graph at checkpoint t+1
#             :param timeout: Maximum computation time (for large graphs)
#             :return: Approximate GED score (lower = more similar)
#             """
#             def custom_edge_match(edge1, edge2):
#                 # Extract 'action' and 'weight' attributes from both edges
#                 a1, w1 = edge1['action'], edge1['weight']
#                 a2, w2 = edge2['action'], edge2['weight']
                
#                 if a1 != a2:
#                     # Return 1 if the actions are different
#                     return 1
#                 else:
#                     # Return the absolute difference of weights if the actions are the same
#                     return abs(w1 - w2)
#             try:
#                 ged =  nx.graph_edit_distance(G1, G2, timeout=timeout, edge_match=custom_edge_match)
#                 ged = ged if ged is not None else float('inf')  # If computation fails, return infinite distance
#             except nx.NetworkXError:
#                 return float('inf')  # If computation fails, return infinite distance
#             max_edges = max(len(G1.edges()), len(G2.edges()))
#             return ged / max_edges if max_edges > 0 else 0
        
#         def compute_custom_wasserstein_distance_graphs(G1, G2, distance_fn, epsilon=1e-8):
#             """
#             Compute the mean Wasserstein Distance between two RL transition graphs using a custom
#             ground distance function.

#             Parameters:
#             -----------
#             G1, G2 : networkx.DiGraph
#                 Directed graphs representing RL transition graphs. They are assumed to have a
#                 normalized 'prob' attribute on each edge.
#             distance_fn : function
#                 A function that takes two nodes as input and returns the distance between them.

#             Returns:
#             --------
#             mean_wd : float
#                 The mean Wasserstein distance computed across all source states (rows).
#             """
#             # 1. Determine the union of nodes from both graphs
#             nodes = sorted(set(G1.nodes()).union(set(G2.nodes())))
#             n = len(nodes)
            
#             # Create a mapping from node to index for consistent ordering.
#             node_to_index = {node: i for i, node in enumerate(nodes)}
            
#             # 2. Construct normalized adjacency matrices for both graphs.
#             # Initialize matrices with zeros.
#             P = np.zeros((n, n))
#             Q = np.zeros((n, n))
            
#             # For graph G1: for each edge, set the corresponding entry.
#             for u, v, data in G1.edges(data=True):
#                 i = node_to_index[u]
#                 j = node_to_index[v]
#                 # Use the 'prob' attribute if it exists, else assume 1.
#                 P[i, j] = data.get("weight", 0)
            
#             # For graph G2:
#             for u, v, data in G2.edges(data=True):
#                 i = node_to_index[u]
#                 j = node_to_index[v]
#                 Q[i, j] = data.get("weight", 0)

#             # 3. Build the custom cost matrix using the distance function.
#             # Cost matrix C where C[i,j] = distance_fn(nodes[i], nodes[j])
#             C = np.zeros((n, n))
#             for i in range(n):
#                 for j in range(n):
#                     C[i, j] = distance_fn(self._id_to_state[nodes[i]], self._id_to_state[nodes[j]])
#             # 4. Compute the Wasserstein distance for each source state (row-wise).
#             row_distances = []
#             for i in range(n):
#                 p = P[i]
#                 q = Q[i]
#                 # If both distributions are all zeros (i.e., the state has no outgoing transitions in both graphs), distance is 0.
#                 if np.sum(p) < epsilon and np.sum(q) < epsilon:
#                     row_distances.append(0)
#                 elif np.sum(p) < epsilon or np.sum(q) < epsilon:
#                     return np.max(C)
#                 else:
#                     wd2 = ot.emd2(p, q, C)
#                     wd = np.sqrt(wd2)
#                     row_distances.append(wd)
    
#             mean_wd = np.mean(row_distances)
#             return mean_wd
        
#         metrics = {}
#         graphs = {i: self.build_graph(i) for i in range(self.num_checkpoints)}
#         for i,j in zip(range(self.num_checkpoints), range(1, self.num_checkpoints)):
#             metrics[i] = {}
#             metrics[i]["jaccard"] = directed_jaccard_similarity(graphs[i], graphs[j])
#             #metrics[i]["jaccard_weighted"] = directed_jaccard_similarity_weighted(graphs[i], graphs[j])
#             #metrics[i]["preferential_attachment"] = directed_preferential_attachment_between_graphs(graphs[i], graphs[j])
#             metrics[i]["overlap_edges"] = graph_overlap_coefficient(graphs[i], graphs[j])
#             #metrics[i]["overlap_nodes"] = graph_overlap_coefficient_nodes(graphs[i], graphs[j])
#             metrics[i]["dice"] = graph_dice_coefficient(graphs[i], graphs[j])
#             metrics[i]["GED"] = compute_graph_edit_distance(graphs[i], graphs[j])
#             metrics[i]["wasserstein"] = compute_custom_wasserstein_distance_graphs(graphs[i], graphs[j], distance_fn=state_dist_fn)
#         return metrics


#     def compute_metric_correlations(self):
#             """
#             Computes the correlation matrix between pairs of metrics generated by
#             get_changes_per_checkpoint, without using pandas.

#             Returns:
#                 A NumPy array representing the correlation matrix.
#             """

#             changes = self.get_changes_per_checkpoint()

#             if not changes:
#                 return None  # Handle empty changes

#             # Extract metric values into a dictionary of lists
#             metric_data = {
#                 "added_nodes": [],
#                 "removed_nodes": [],
#                 "added_edges": [],
#                 "removed_edges": [],
#                 "added_self_loops": [],
#                 "removed_self_loops": [],
#                 "added_simple_edges": [],
#                 "removed_simple_edges": [],
#                 "added_simple_loops": [],
#                 "removed_simple_loops": [],
#                 "winrate_change": [],
#                 "steps_change": [],
#             }

#             for checkpoint_data in changes.values():
#                 for metric, value in checkpoint_data.items():
#                     metric_data[metric].append(value)

#             # Convert dictionary of lists to NumPy array
#             metric_names = list(metric_data.keys())
#             num_metrics = len(metric_names)
#             num_checkpoints = len(list(changes.values())[0].values())

#             metric_array = np.zeros((num_metrics, num_checkpoints))
#             for i, metric_name in enumerate(metric_names):
#                 metric_array[i, :] = metric_data[metric_name]

#             # Compute the correlation matrix using NumPy
#             correlation_matrix = np.corrcoef(metric_array)

#             return correlation_matrix

if __name__ == '__main__':

    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--end_reason", help="Filter options for trajectories", default=None, type=str, action='store', required=False)
    # parser.add_argument("--n_trajectories", help="Limit of how many trajectories to use", action='store', default=5000, required=False)
    
    # args = parser.parse_args()
       
   
    # # Experiment 007
    # # tg_blocks.add_checkpoint(read_json("./trajectories/2025-01-30_SARSAAgent_Attacker_experimentsarsa_007_coordinatorV3-episodes-1000.jsonl",max_lines=args.n_trajectories))
    # # tg_blocks.add_checkpoint(read_json("./trajectories/2025-01-30_SARSAAgent_Attacker_experimentsarsa_007_coordinatorV3-episodes-1500.jsonl",max_lines=args.n_trajectories))
    # # tg_blocks.add_checkpoint(read_json("./trajectories/2025-01-30_SARSAAgent_Attacker_experimentsarsa_007_coordinatorV3-episodes-2000.jsonl",max_lines=args.n_trajectories))
    # # tg_blocks.add_checkpoint(read_json("./trajectories/2025-01-30_SARSAAgent_Attacker_experimentsarsa_007_coordinatorV3-episodes-2500.jsonl",max_lines=args.n_trajectories))
    # # tg_blocks.add_checkpoint(read_json("./trajectories/2025-01-30_SARSAAgent_Attacker_experimentsarsa_007_coordinatorV3-episodes-4500.jsonl",max_lines=args.n_trajectories))
    # #tg_blocks.save_to_pickle("/data/ondra/aidojo/trajectories/experiment_007.pickle")
    
    # # Experiment 005
    # # SARSA 005
    # # tg_005_sarsa = TrajectoryGraph.load_from_pickle("/data/ondra/aidojo/trajectories/tg_experiment_005_sarsa.pickle")
    # # tg_005_sarsa.plot_tg_stats_per_checkpoint(fig_sufix="sarsa_005")
    # # print(tg_005_sarsa.get_graph_metrics_per_checkpoint())
    # # tg_005_sarsa.plot_graph_metrics_per_checkpoint(fig_sufix="sarsa_005")
    # tg_009_sarsa = TrajectoryGraph.load_from_pickle("/data/ondra/aidojo/trajectories/experiment_009_test.pickle")
    # tg_009_sarsa.plot_tg_stats_per_checkpoint(fig_sufix="sarsa_009_test")
    # tg_009_sarsa.plot_graph_metrics_per_checkpoint(fig_sufix="sarsa_009_test")

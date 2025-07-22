#from AIDojoCoordinator.utils.trajectory_analysis import read_json
import numpy as np
import os 
#import AIDojoCoordinator.utils.utils as utils
import argparse
import matplotlib.pyplot as plt
import pickle
#from scipy.stats import pearsonr
import pandas as pd
# import ot
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
from utils.trajectory_utils import compute_lambda_returns
# from scipy.spatial.distance import mahalanobis
# from scipy.linalg import inv
# from ruptures.base import BaseCost
import networkx as nx
from typing import Iterable
from trajectory import Trajectory, Transition

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

def plot_tg_mdp(graph, filename, node_clusters=None):
    """
    Plots the transition graph of a Markov Decision Process (MDP) using NetworkX and highlights clusters.

    Args:
        graph: The transition graph of the MDP.
        filename: Output filename for the rendered graph (e.g., 'mdp.png').
        node_labels: Optional. Dict mapping node_id -> list of cluster_ids.
    """
    G = nx.MultiDiGraph()
    for (s1, s2, a), freq in graph.get_probability_per_edge().items():
        G.add_edge(s1, s2, weight=freq, action=a)

    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update({
        "rankdir": "LR",
        "nodesep": "0.2",
        "ranksep": "0.3",
        "splines": "true",
    })

    # Add edge labels
    for u, v, k, d in G.edges(data=True, keys=True):
        label = f"{d['action']} ({d['weight']:.2f})"
        A.get_edge(u, v, k).attr['label'] = label

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
            node.attr['fillcolor'] = 'white'

    # Optional: Add subgraphs for clusters
    if node_clusters:
        for cluster_id, nodes in node_clusters.items():
            sg = A.add_subgraph(
                list(nodes),
                name=f"cluster_{cluster_id}",
                label=f"Cluster {cluster_id}",
                style="dotted",
                color="orange",
                fontsize="10",
                fontcolor="black"
            )
            sg.graph_attr.update({
                "margin": "15",       # Increase space inside cluster boundary (default is 6–8)
                "penwidth": "5",      # Thicker dotted line
            })

    A.layout(prog='dot')
    A.draw(filename)

def get_trajectory_rewards(trajectory:list):
    return [x.reward for x in trajectory]

def get_trajectory_action_surprises(trajectory:Iterable, graph, epsilon=1e-8):
    """
    Computes the action surprise of a trajectory.
    """
    action_surprises = []
    for transition in trajectory:
        action_surprise = graph.compute_action_surprise(transition, epsilon)
        action_surprises.append(action_surprise)
    return action_surprises


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
        self._src_state_count = {} # counter of how many times state was a source state
        self._policy = {}
        self._returns = []
        self._lengths = []
        self._starting_states = set()
        self._terminal_states = set()
    
    def update_policy(self, transition:Transition)->None:
        """
        Updates the policy of the graph.
        """
        state = transition.state
        if state not in self._policy:
            self._policy[state] = {}
        if transition.action not in self._policy[state]:
            self._policy[state][transition.action] = 0
        self._policy[state][transition.action] += 1
    
    def get_empirical_policy_in_state(self, state)->dict:
        """
        Returns empirical probability of actions in a given state.
        """
        return self._policy.get(state, None)
    
    def get_action_empirical_probability(self, state, action)->float:
        """
        Returns empirical probability of taking action in a given state.
        """
        if state not in self._policy:
            return 0
        if action not in self._policy[state]:
            return 0
        return self._policy[state][action]/sum(self._policy[state].values())

    @property
    def num_trajectories(self)->int:
        return len(self._returns)

    def update_edge_reward(self, edge_id:tuple, reward)->None:
        if edge_id not in self._edge_reward:
            self._edge_reward
        else:
            self._edge_reward[edge_id] += (reward - self._edge_reward[edge_id])/self._edge_count[edge_id]
    

    def add_trajectory(self, trajectory:Trajectory)->None:
        trajectory_return = 0
        self._starting_states.add(trajectory[0].state)
        for transition in trajectory:
            src_id = transition.state
            action_id = transition.action
            dst_id =transition.next_state
            reward = transition.reward
            self.update_policy(transition)
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
        # for (s,a,d) in self._edge_count.keys():
        #     if isinstance(s, np.array):
        #         if np.array_equal(s, d):
        #             loops.add((s,a,d))
        #     else:
        #         if s == d:
        #             loops.add((s,a,d))
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

    def compute_action_surprise(self, transition:Transition, epsilon=1e-8):
        """
        Computes the surprise of taking an action in a given state.
        """
        action_surprise = -np.log(self.get_action_empirical_probability(transition.state, transition.action) + epsilon)
        return action_surprise
    
    def compute_trajectory_set_action_surprise(self, trajectories:Iterable, epsilon=1e-8):
        """
        Computes the surprise of taking an action in a given state.
        """
        action_surprises = []
        for trajectory in trajectories:
            for transition in trajectory:
                action_surprise = self.compute_action_surprise(transition, epsilon)
                action_surprises.append(action_surprise)
        return np.mean(action_surprises)
    
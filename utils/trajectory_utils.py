# Author: Ondrej Lukas, ondrej.lukas@aic.fel.cvut.cz
import numpy as np
import ruptures as rpt
from typing import Iterable
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from typing import List, Optional, Callable, Any
from trajectory import Transition, Trajectory, Policy, EmpiricalPolicy
import networkx as nx


def compute_kl_divergence(state: Any, policy1:EmpiricalPolicy, policy2:EmpiricalPolicy, num_actions:int, alpha=1.0, epsilon=1e-8) -> float:
    """
    Compute the KL divergence between two empirical policies at a given state:
    KL[policy_new || policy_old]

    Parameters:
        state: any hashable state representation
        policy1: empirical policy object with .get_action_probability(state, action, alpha)
        policy2: another empirical policy object with the same API
        num_actions: total number of discrete actions
        alpha: Laplace smoothing constant
        epsilon: small value to prevent log(0)

    Returns:
        float: KL divergence value
    """
    kl = 0.0

    for action in range(num_actions):
        p = policy1.get_action_probability(state, action, alpha)
        q = policy2.get_action_probability(state, action, alpha)

        # Clip to avoid log(0)
        p = max(p, epsilon)
        q = max(q, epsilon)

        kl += p * (np.log(p) - np.log(q))
    return kl

def compute_js_divergence(state: Any, policy1: EmpiricalPolicy, policy2: EmpiricalPolicy,
                          num_actions: int, alpha=1.0, epsilon=1e-8) -> float:
    """
    Compute the Jensen–Shannon (JS) divergence between two empirical policies at a given state.
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)

    Parameters:
        state: any hashable state representation
        policy1: empirical policy object with .get_action_probability(state, action, alpha)
        policy2: another empirical policy object with the same API
        num_actions: total number of discrete actions
        alpha: Laplace smoothing constant
        epsilon: small value to prevent log(0)

    Returns:
        float: JS divergence value
    """
    kl_p_m = 0.0
    kl_q_m = 0.0

    for action in range(num_actions):
        p = policy1.get_action_probability(state, action, alpha)
        q = policy2.get_action_probability(state, action, alpha)

        # Clip to avoid log(0)
        p = max(p, epsilon)
        q = max(q, epsilon)

        m = 0.5 * (p + q)
        m = max(m, epsilon)

        kl_p_m += p * (np.log(p) - np.log(m))
        kl_q_m += q * (np.log(q) - np.log(m))

    js = 0.5 * kl_p_m + 0.5 * kl_q_m
    return js

def compute_normalized_surprise(state, action, policy_new, policy_old, num_actions, alpha=0.1, epsilon=1e-8):
    # Get smoothed probabilities
    p_new = max(policy_new.get_action_probability(state, action, alpha), epsilon)
    p_old = max(policy_old.get_action_probability(state, action, alpha), epsilon)

    # Log-prob difference
    log_diff = np.log(p_new) - np.log(p_old)
    # KL divergence at state
    kl = compute_kl_divergence(state, policy_new, policy_old, num_actions, alpha, epsilon)
    js = compute_js_divergence(state, policy_new, policy_old, num_actions, alpha, epsilon)

    # Normalize
    return log_diff / max(js, epsilon)

def compute_trajectory_surprises(trajectory:Trajectory, policy:Policy, previous_policy:Policy, epsilon=1e-8)->List[float]:
    """
    Computes the surprise of a trajectory given a policy.
    Args:
        trajectory (Trajectory): A trajectory, which is a list of transitions.
        policy (Policy): A policy that defines the action probabilities.
    Returns:
        List[float]: A list of surprises for each transition in the trajectory.
    """
    surprises = []
    for transition in trajectory:
        # action_prob = policy.get_action_probability(transition.state, transition.action)
        # prev_action_prob = previous_policy.get_action_probability(transition.state, transition.action)
        # surprise = np.log(action_prob) - np.log(prev_action_prob+epsilon) 
        surprise = compute_normalized_surprise(transition.state, transition.action, policy, previous_policy, policy.num_actions)
        surprises.append(surprise)
    return surprises

def build_graph(policy:EmpiricalPolicy)->nx.DiGraph:
    G = nx.DiGraph()
    for (state, action, next_state) ,count in policy._edge_count.items():
          G.add_edge(
            state,
            next_state,
            action=action,
            count=count,
            reward=policy._edge_reward.get((state, action, next_state), 0)
        )
    return G

def empirical_policy_statistics(policy:EmpiricalPolicy, is_win_fn:Optional[Callable[[Trajectory], bool]] = lambda x: len(x) > 0 and x[-1].reward > 0)->dict:
    """
    Computes the statistics for a given Empirical policy.
    Winrate is determined based on a provided function.
    Args:
    Returns:

    """
    policy_graph = build_graph(policy)
    metrics = {}
    # size of the empirical state space
    metrics["unique_nodes"] = policy.num_states
    metrics["unique_actions"] = policy.num_actions
    metrics["mean_trajectory_lenght"] = np.mean([len(t) for t in policy.trajectories])
    metrics["mean_return"] = np.mean([np.sum(t.rewards) for t in policy.trajectories])
    metrics["mean_winrate"] = sum([is_win_fn(t) for t in policy.trajectories])/policy.num_trajectories
    metrics["loops"] = nx.number_of_selfloops(policy_graph)
    metrics["unique_edges"] = len(policy_graph.edges)
    return metrics

def compute_lambda_returns(trajectory:Trajectory, gamma=0.99, lam=0.95)->np.ndarray:
    """
    Compute the lambda returns for a trajectory.
    Args:
        trajectory (Iterable): A trajectory, which is a list of transitions.
        gamma (float): Discount factor.
        lam (float): Lambda for eligibility traces.
    Returns:
        np.ndarray: Lambda returns for the trajectory (np.array).
    """
    T = len(trajectory)
    λ_ret = np.zeros(T)
    λ_ret[-1] = trajectory[-1].reward
    for t in reversed(range(T - 1)):
        λ_ret[t] = trajectory[t].reward + gamma * lam * λ_ret[t + 1]
    return λ_ret

def compute_eligibility_traces_sa(trajectory: Trajectory, gamma=0.99, lam=0.95):
    e = defaultdict(float)
    traces = []

    for transition in trajectory:
        s, a = transition.state, transition.action

        # Decay all traces
        for key in e:
            e[key] *= gamma * lam

        # Bump trace for current state-action
        e[(s, a)] += 1.0

        # Store a snapshot
        traces.append(dict(e))  # shallow copy
    # Prune low values?
    return traces  # List of dicts: one per timestep

def compute_credit_per_step(trajectory:Trajectory, gamma=0.99, lam=0.95):
    """
    Compute credit per step for a trajectory.
    Args:
        trajectory (Iterable): A trajectory, which is a list of transitions.
        gamma (float): Discount factor.
        lam (float): Lambda for eligibility traces.
    Returns:
        np.ndarray: Credit per step for the trajectory (np.array).
    """
    value = compute_lambda_returns(trajectory, gamma, lam)
    traces = compute_eligibility_traces_sa(trajectory, gamma, lam)
    credit = defaultdict(float)                   # final attribution per (s,a)
    for t, e_t in enumerate(traces):              # iterate over timesteps
        v = value[t]                              # scalar to propagate at step t
        for (s,a), elig in e_t.items():           # sparse over only active pairs
            credit[(s,a)] += elig * v 
  


def find_trajectory_segments(trajectory:Trajectory, policy:Policy, previous_policy:Policy, penalty=2, trajectory_id=None)->List[dict]:
    """
    """
    rewards = np.array(trajectory.rewards)
    surprises = np.array(compute_trajectory_surprises(trajectory, policy, previous_policy, epsilon=1e-12))
    lambda_returns = np.array(compute_lambda_returns(trajectory))
    features = np.stack([lambda_returns, surprises, rewards]).T # should be of shape (len(trajectory), num_features)
    features = StandardScaler().fit_transform(features) 
    algo = rpt.KernelCPD(kernel="rbf", min_size=2).fit(features)
    trajectory_segments = []
    try:
        break_points = algo.predict(pen=penalty)
        break_points = [0] + break_points  # prepend 0
        segments_idx = list(zip(break_points[:-1], break_points[1:]))
        for (start, end) in segments_idx:
            if start==end:
                continue
            seg = {
                "start": start,
                "end": end,
                "features": get_segment_features(start, end, surprises, rewards, lambda_returns, trajectory),
            }
            if trajectory_id is not None:
                seg["trajectory_id"] = trajectory_id
            trajectory_segments.append(seg)
    except rpt.exceptions.BadSegmentationParameters:
        pass
    return trajectory_segments

def get_segment_features(seg_start:int, seg_end:int ,surprises:np.ndarray,rewards:np.ndarray, elegibility_traces:np.ndarray, trajectory:Trajectory):
    """
    Computes the features for a segment.
    """
    features = {}
    features["et"] = np.mean(elegibility_traces[seg_start:seg_end])
    features["et_std"] = np.std(elegibility_traces[seg_start:seg_end])
    features["surprise"] = np.mean(surprises[seg_start:seg_end])
    features["surprise_std"] = np.std(surprises[seg_start:seg_end])
    features["reward"] = np.mean(rewards[seg_start:seg_end])
    features["reward_std"] = np.std(rewards[seg_start:seg_end])
    features["coverage"] = (seg_end - seg_start)/len(trajectory)
    features["pos_start"] = seg_start/len(trajectory)
    features["pos_end"] = seg_end/len(trajectory)
    return features

def get_cluster_features(segments:Iterable):
    features = np.zeros([len(segments), len(segments[0][1].values())]) 
    for i,s in enumerate(segments):
        for j,x in enumerate(s[1].values()):
            features[i][j] = x
    ret = {}
    for i, k in enumerate(segments[0][1].keys()):
        ret[k] = {
            "mean":np.mean(features[:, i]),
            "std":np.std(features[:, i]),
            # "min":np.min(features[:, i]),
            # "max":np.max(features[:, i]),
            # "median":np.median(features[:, i]),
        }
    return ret
 
def cluster_segments(segments:Iterable):
    features = [list(s["features"].values()) for s in segments]
    clustering = DBSCAN(eps=5, min_samples=2).fit(features)
    clusters ={}
    for segment, cluster_id in zip(segments, clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(segment)
    return clusters

def get_motifs(trajectories, graph, epsilon=1e-12, penalty=2):
    """
    Computes the motifs in the transition graph.
    """
    raise DeprecationWarning("get_motifs is deprecated, use find_trajectory_segments instead")
    motifs = {}
    motfif_candidates = []
    for t in trajectories:
        rewards = np.array([step.reward for step in t])
        lambda_ret = compute_lambda_returns(t)
        surprises = np.array(get_trajectory_action_surprises(t, graph, epsilon))
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(np.stack([lambda_ret, surprises, rewards], axis=1))
        algo = rpt.KernelCPD(kernel="rbf", min_size=1).fit(features_scaled)
        bkpts = algo.predict(pen=penalty)
        segments = [(0, bkpts[0])] + [(bkpts[i], bkpts[i+1]) for i in range(len(bkpts)-1)]
        for start, end in segments:
            m = tuple(t[start:end])
            motfif_candidates.append((m, get_segment_features(start, end, surprises, rewards, lambda_ret, t)))
            if m not in motifs:
                motifs[m] = 0
            motifs[m] += 1
        # add segment occurence to the motif features
        final_candidates = []
        for m_candidate, features in motfif_candidates:
            features["occurences"] = motifs[m]
            final_candidates.append((m_candidate, features))
    clusters = cluster_segments(final_candidates)
    return motifs, clusters

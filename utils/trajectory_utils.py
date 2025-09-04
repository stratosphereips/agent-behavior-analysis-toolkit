# Author: Ondrej Lukas, ondrej.lukas@aic.fel.cvut.cz
import json
import numpy as np
import ruptures as rpt
from typing import Iterable
from sklearn.preprocessing import StandardScaler, RobustScaler
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from typing import List, Optional, Callable, Any
from trajectory import Transition, Trajectory, Policy, EmpiricalPolicy
import networkx as nx
import json
import os
from utils.aidojo_utils import aidojo_rebuild_trajectory
import hdbscan
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

###### ENCODERS ######
def numpy_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj
######################

def store_trajectories_to_json(trajectory_set:Iterable, filename:str, metadata:dict=None, encoder=None) -> None:
    """
    Store a set of trajectories to a JSON file.
    """
    json_data = {
        "trajectories": [traj.to_json(metadata) for traj in trajectory_set]
    }
    if metadata:
        json_data["metadata"] = metadata
        print(metadata)
    with open(filename, 'w') as f:
        if encoder:
            json.dump(json_data, f, default=encoder)
        else:
            json.dump(json_data, f)

def load_trajectories_from_json(filename: str, load_metadata: bool=False, max_trajectories: int=None) -> tuple[Iterable[Trajectory], dict]:
    """
    Load a set of trajectories from a JSON file.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".jsonl":
        return load_trajectories_from_jsonl(filename, load_metadata, max_trajectories)
    else:
        with open(filename, 'r') as f:
            json_data = json.load(f)
        trajectories = [Trajectory.from_json(traj) for traj in json_data.get("trajectories", [])]
        if max_trajectories:
            trajectories = trajectories[:max_trajectories]

        metadata = json_data.get("metadata", {}) if load_metadata else {}
        return trajectories, metadata

def load_trajectories_from_jsonl(
    filename: str, 
    load_metadata: bool = False, 
    max_trajectories: int = None,
) -> tuple[Iterable["Trajectory"], dict]:
    """
    Load a set of trajectories from a JSONL file (one JSON object per line).
    Each line corresponds to a trajectory object.
    """
    print(f"\tLoading trajectories from {filename}")
    trajectories = []
    metadata = {}

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if max_trajectories and i >= max_trajectories:
                break
            obj = json.loads(line)

            # Handle metadata if present
            if load_metadata:
                if "metadata" in obj:
                    metadata.update(obj["metadata"])
                else:
                    metadata.update({k: v for k, v in obj.items() if k != "trajectory"})

            try:
                traj = obj["trajectory"]
                states = traj.get("states", None)
                actions = traj.get("actions", None)
                rewards = traj.get("rewards", None)
                trajectories.append(aidojo_rebuild_trajectory(states, actions, rewards))
            except KeyError as e:
                print(f"Error loading trajectory from line {i}: {e}")
    return trajectories, metadata

def calculate_ecdf_auc(returns: np.ndarray) -> float:
    """
    Calculate the area under the empirical cumulative distribution function (ECDF)
    of the given returns.
    """
    if returns.size < 2:
        return 0.0

    sorted_returns = np.sort(returns)
    n = sorted_returns.size

    # Widths between consecutive sorted returns
    widths = np.diff(sorted_returns)
    # Heights of ECDF on each interval [x_i, x_{i+1})
    heights = np.arange(1, n) / n  

    # Vectorized dot product = sum(heights * widths)
    auc = np.dot(heights, widths)

    return float(auc)

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
    """
    p = np.array([max(policy1.get_action_probability(state, a, alpha), epsilon) for a in range(num_actions)])
    q = np.array([max(policy2.get_action_probability(state, a, alpha), epsilon) for a in range(num_actions)])
    m = np.maximum(0.5 * (p + q), epsilon)
    kl_p_m = np.sum(p * (np.log(p) - np.log(m)))
    kl_q_m = np.sum(q * (np.log(q) - np.log(m)))
    return 0.5 * (kl_p_m + kl_q_m)

def compute_normalized_surprise(state, action, policy_new, policy_old, per_state_normalization, alpha=0.1, epsilon=1e-8):
    # Get smoothed probabilities
    p_new = policy_new.get_action_probability(state, action, alpha)
    p_old = policy_old.get_action_probability(state, action, alpha)

    # Clip probabilities once
    p_new = max(p_new, epsilon)
    p_old = max(p_old, epsilon)

    # Log-prob difference
    log_diff = np.log(p_new) - np.log(p_old)

    # Use provided JS divergence for this state
    js = per_state_normalization.get(state, epsilon)

    # Normalize
    return log_diff / max(js, epsilon)

def compute_trajectory_surprises(trajectory:Trajectory, policy:Policy, previous_policy:Policy, per_state_normalization:dict, epsilon=1e-8) -> List[float]:
    """
    Computes the surprise of a trajectory given a policy.
    Args:
        trajectory (Trajectory): A trajectory, which is a list of transitions.
        policy (Policy): A policy that defines the action probabilities.
        js_divergence_dict (dict): Dictionary mapping state to JS divergence.
    Returns:
        List[float]: A list of surprises for each transition in the trajectory.
    """
    surprises = []
    for transition in trajectory:
        surprise = compute_normalized_surprise(
            transition.state,
            transition.action,
            policy,
            previous_policy,
            per_state_normalization
        )
        surprises.append(surprise)
    return surprises

def get_trajectory_action_change(trajectory, policy, previous_policy):
    action_changes = []
    for transition in trajectory:
        previous_policy_action = np.argmax([previous_policy.get_action_probability(transition.state, a) for a in range(previous_policy.num_actions)])
        if transition.action == previous_policy_action:
            action_changes.append(0)
        else:
            action_changes.append(1)
    return action_changes

def empirical_policy_statistics(policy:EmpiricalPolicy, is_win_fn:Optional[Callable[[Trajectory], bool]] = lambda x: len(x) > 0 and x[-1].reward > 0)->dict:
    """
    Computes the statistics for a given Empirical policy.
    Winrate is determined based on a provided function.
    Args:
    Returns:
    """
    metrics = {}
    # static metrics
    metrics["unique_nodes"] = policy.num_states
    metrics["unique_actions"] = policy.num_actions
    metrics["mean_trajectory_length"] = np.mean([len(t) for t in policy.trajectories])
    metrics["mean_return"] = np.mean([np.sum(t.rewards) for t in policy.trajectories])
    metrics["return_ecdf_auc"] = calculate_ecdf_auc(np.array([np.sum(t.rewards) for t in policy.trajectories]))
    metrics["mean_winrate"] = sum([is_win_fn(t) for t in policy.trajectories])/policy.num_trajectories

    # Compute self-loops directly from edge counts
    self_loops = sum(1 for (state, action, next_state) in policy._edge_count if state == next_state and next_state is not None)
    metrics["loops"] = self_loops

    # Unique edges: count unique (state, next_state) pairs
    unique_edges = set((state, next_state) for (state, action, next_state) in policy._edge_count if next_state is not None)
    metrics["unique_edges"] = len(unique_edges)
    return metrics

def compute_lambda_returns(rewards: np.ndarray, gamma=0.99, lam=0.95) -> np.ndarray:
    """
    Computes the lambda returns for a trajectory.
    """
    T = len(rewards)
    λ_ret = np.zeros(T)
    λ_ret[-1] = rewards[-1]
    for t in reversed(range(T - 1)):
        λ_ret[t] = rewards[t] + gamma * lam * λ_ret[t + 1]
    return λ_ret

def find_trajectory_segments(
    surprises: np.ndarray,
    rewards: np.ndarray,
    lambda_returns: np.ndarray,
    penalty=5,
    trajectory_id=None,
) -> List[dict]:
    """
    Segment a trajectory using change point detection on standardized features.
    """
    # Stack features efficiently (avoid .T, use axis=1)
    features = np.column_stack((lambda_returns, surprises, rewards))
    trajectory_len = features.shape[0]

    # Standardize features in-place
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # KernelCPD setup
    min_size = max(3, trajectory_len // 20)
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_size, params={"gamma": 0.3})
    algo.fit(features)

    trajectory_segments = []
    try:
        break_points = algo.predict(pen=penalty)
        break_points = [0] + break_points
        for start, end in zip(break_points[:-1], break_points[1:]):
            if start == end:
                continue
            seg = {
                "start": start,
                "end": end,
                "features": tuple(get_segment_features(start, end, surprises, rewards, lambda_returns, trajectory_len).values()),
                "surprises": surprises[start:end],
            }
            if trajectory_id is not None:
                seg["trajectory_id"] = trajectory_id
            trajectory_segments.append(seg)
    except rpt.exceptions.BadSegmentationParameters:
        pass
    return trajectory_segments

def get_segment_features(seg_start:int, seg_end:int ,surprises:np.ndarray,rewards:np.ndarray, elegibility_traces:np.ndarray, trajectory_len:int):
    """
    Computes the features for a segment.
    """
    feature_names = ["λ_ret", "λ_ret_std", "surprise", "surprise_std", "reward", "reward_std", "length", "pos_start", "pos_end"]
    features = {}
    features["λ_ret"] = np.mean(elegibility_traces[seg_start:seg_end])
    features["λ_ret_std"] = np.std(elegibility_traces[seg_start:seg_end])
    features["surprise"] = np.mean(surprises[seg_start:seg_end])
    features["surprise_std"] = np.std(surprises[seg_start:seg_end])
    features["reward"] = np.mean(rewards[seg_start:seg_end])
    features["reward_std"] = np.std(rewards[seg_start:seg_end])
    features["length"] = (seg_end - seg_start)
    features["pos_start"] = seg_start
    features["pos_end"] = seg_end
    return features
 
# def cluster_segments(segments: Iterable):
#     # Extract features as a numpy array for efficient clustering
#     features = np.array([s["features"] for s in segments])
#     # Use a reasonable min_samples value (at least 2, or based on feature count)
#     min_samples = max(2, len(features[0]) + 1)
#     clustering = DBSCAN(eps=5, min_samples=min_samples).fit(features)
#     # Use defaultdict for faster cluster assignment
#     clusters = defaultdict(list)
#     for segment, cluster_id in zip(segments, clustering.labels_):
#         clusters[cluster_id].append(segment)
#     return dict(clusters)
def cluster_segments(
    segments,
    include_features=None,
    eps=1.2,
    min_samples=None,
    scale=True,
):
    """
    Cluster trajectory segments based on their features.

    Parameters:
        segments: list of dicts, each with a "features" key containing a feature dict
        include_features: list of feature names to include (default = all)
        eps: DBSCAN eps parameter (scale-dependent!)
        min_samples: DBSCAN min_samples parameter (default = len(features)+1)
        scale: whether to standardize features before clustering

    Returns:
        clusters: dict mapping cluster_id -> list of segments
    """

    # # --- Select features ---
    # if include_features is None:
    #     # use all feature keys from first segment
    #     include_features = list(segments[0]["features"].keys())

    X = np.array([s["features"] for s in segments])

    # --- Normalize ---
    if scale:
        X = StandardScaler().fit_transform(X)
    # --- DBSCAN parameters ---
    if min_samples is None:
        min_samples = max(5, X.shape[1] + 1)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    # --- Collect results ---
    clusters = defaultdict(list)
    for segment, cluster_id in zip(segments, clustering.labels_):
        clusters[cluster_id].append(segment)

    return dict(clusters)

def get_clusters_per_step(trajectory, clusters)->list:
    clusters_per_step = defaultdict(list)
    for cluster_id, segments in clusters.items():
        for segment in segments:
            for step in range(segment["start"], segment["end"]):
                clusters_per_step[step].append(cluster_id)
    return [int(list(set(clusters_per_step[i]))[0]) for i in range(len(trajectory))]

def js_divergence_per_state(policy_p: EmpiricalPolicy, policy_q: EmpiricalPolicy, alpha: float = 0.1):
    states = set(policy_p._state_action_map.keys()) | set(policy_q._state_action_map.keys())
    if not states:
        return {}, 0.0  # empty dict and mean

    global_actions = set()
    for amap in policy_p._state_action_map.values():
        global_actions.update(amap.keys())
    for amap in policy_q._state_action_map.values():
        global_actions.update(amap.keys())
    global_actions = list(global_actions)

    js_per_state = {}
    for state in states:
        p_probs = np.array([policy_p.get_action_probability(state, a, alpha) for a in global_actions])
        q_probs = np.array([policy_q.get_action_probability(state, a, alpha) for a in global_actions])
        m_probs = 0.5 * (p_probs + q_probs)

        kl_pm = np.sum(p_probs * np.log(p_probs / m_probs))
        kl_qm = np.sum(q_probs * np.log(q_probs / m_probs))
        js_per_state[state] = 0.5 * (kl_pm + kl_qm)

    mean_js = float(np.mean(list(js_per_state.values())))
    return js_per_state, mean_js

    

def policy_comparison(curr_policy:EmpiricalPolicy, prev_policy:EmpiricalPolicy)->dict:
    """
    Compare two policies based on their trajectory statistics.
    """
    per_state_js_div, mean_js_div = js_divergence_per_state(curr_policy, prev_policy)

    metrics = {
        "node_overlap": len(set(curr_policy.states) & set(prev_policy.states))/max(len(curr_policy.states), len(prev_policy.states), 1),
        "edge_overlap": len(set(curr_policy.actions) & set(prev_policy.actions))/max(len(curr_policy.actions), len(prev_policy.actions), 1),
        "js_divergence": mean_js_div
    }
    return metrics, per_state_js_div
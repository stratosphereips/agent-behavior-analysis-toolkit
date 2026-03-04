# Author: Ondrej Lukas, ondrej.lukas@aic.fel.cvut.cz
import json
import numpy as np
import ruptures as rpt
from typing import Iterable
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict
from typing import List, Optional, Callable, Any
from trajectory import Transition, Trajectory, Policy, EmpiricalPolicy
import networkx as nx
import json
import os
from utils.aidojo_utils import aidojo_rebuild_trajectory, aidojo_action_type_from_dict, aidojo_state_str_from_dict
from ruptures import costs
from typing import Dict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
import collections
from scipy.optimize import linear_sum_assignment
import random

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

def load_trajectories_from_json(
    filename: str,
    load_metadata: bool=False,
    max_trajectories: int|None=None,
    action_encoder:Callable|None=None,
    state_encoder:Callable|None=None) -> tuple[List[Trajectory], dict]:
    """
    Load a set of trajectories from a JSON file.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".jsonl":
        print("JSONL file detected, using JSONL loader")
        return load_trajectories_from_jsonl(filename, load_metadata, max_trajectories, action_encoder, state_encoder)
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
    max_trajectories: int | None = None,
    action_encoder: Callable | None = None,
    state_encoder: Callable | None = None
) -> tuple[List["Trajectory"], dict]:
    """
    Load a set of trajectories from a JSONL file (one JSON object per line).
    Each line corresponds to a trajectory object.
    """
    print(f"\tLoading trajectories from {filename}")
    trajectories = []
    metadata = {}

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if max_trajectories and len(trajectories) >= max_trajectories:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i}: {e}")
                continue
            
            # Handle metadata if present
            if load_metadata:
                if "metadata" in obj:
                    metadata.update(obj["metadata"])
                else:
                    metadata.update({k: v for k, v in obj.items() if k != "trajectory"})

            try:
                traj_data = obj.get("trajectory")
                if traj_data:
                    states = traj_data.get("states", None)
                    actions = traj_data.get("actions", None)
                    rewards = traj_data.get("rewards", None)
                    reconstructed_trajectory = rebuild_trajectory_from_components(
                        states,
                        actions,
                        rewards,
                        action_encoder=action_encoder,
                        state_encoder=state_encoder
                    )
                    # print(reconstructed_trajectory.total_reward)
                    trajectories.append(reconstructed_trajectory)
            except Exception as e:
                print(f"Error processing trajectory from line {i}: {e}")
    print(f"Number of trajectories: {len(trajectories)}")
    return trajectories, metadata

def rebuild_trajectory_from_components(
    states: list,
    actions: Iterable,
    rewards: Iterable,
    action_encoder: Callable | None = None,
    state_encoder: Callable | None = None
) -> Trajectory:
    """
    Rebuild a Trajectory object from its components.
    """
    traj = Trajectory()
    for s, a, r, s_next in zip(states, actions, rewards, states[1:]):
        if state_encoder:
            s = state_encoder(s)
            s_next = state_encoder(s_next)
        if action_encoder:
            a = action_encoder(a)
        traj.add_transition(s, a, r, s_next)
    return traj

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

def compute_normalized_surprise(state, action, policy_new:Policy, policy_old:Policy, per_state_normalization:dict, alpha=0.1, epsilon=1e-8):
    """
    Computes the surprise of an action given a policy change, normalized by the JS divergence of the state's policy change.
    
    Surprise(s, a) = (log(P_new(a|s)) - log(P_old(a|s))) / JS(P_new(.|s) || P_old(.|s))
    
    """
    # Get smoothed probabilities
    p_new = policy_new.get_action_probability(state, action, alpha)
    p_old = policy_old.get_action_probability(state, action, alpha)

    # Clip probabilities once
    p_new = max(p_new, epsilon)
    p_old = max(p_old, epsilon)

    # Log-prob difference
    #(log(P_new) - log(P_old)) = log(P_new / P_old)
    # If P_new > P_old, this is positive (we are surprised by how much more likely it is now)
    # If P_new < P_old, this is negative (we are surprised by how much less likely it is now)
    
    # Wait, "Surprise" usually means -log(P). 
    # Here we are looking for "Surprise relative to previous model".
    # If the model thinks an action is probable, and previous thought it was rare => High Positive Shift?
    
    # Let's stick to the log-ratio for now as a "Shift" metric.
    log_diff = np.log(p_new) - np.log(p_old)

    # Use provided JS divergence for this state
    js = per_state_normalization.get(state, epsilon)

    # Normalize
    # We add epsilon to JS to avoid division by zero if policies are identical
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

# def find_trajectory_segments(
#     surprises: np.ndarray,
#     rewards: np.ndarray,
#     lambda_returns: np.ndarray,
#     actions: np.ndarray,
#     states: np.ndarray,
#     penalty=5,
#     trajectory_id=None,
# ) -> List[dict]:
#     """
#     Segment a trajectory using change point detection on standardized features.
#     """
#     # Stack features efficiently (avoid .T, use axis=1)
#     features = np.column_stack((lambda_returns, surprises, rewards))
#     trajectory_len = features.shape[0]

#     # Standardize features in-place
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)

#     # KernelCPD setup
#     min_size = max(3, trajectory_len // 20) #????
#     algo = rpt.KernelCPD(kernel="rbf", min_size=min_size, params={"gamma": 0.3})
#     algo.fit(features)

#     trajectory_segments = []
#     try:
#         break_points = algo.predict(pen=penalty)
#         break_points = [0] + break_points
#         for start, end in zip(break_points[:-1], break_points[1:]):
#             if start == end:
#                 continue
#             seg = {
#                 "start": start,
#                 "end": end,
#                 "features": tuple(get_segment_features(start, end, surprises, rewards, lambda_returns, actions, states, trajectory_len).values()),
#                 "surprises": surprises[start:end],
#                 "return": np.mean(rewards[start:end]),
#             }
#             if trajectory_id is not None:
#                 seg["trajectory_id"] = trajectory_id
#             trajectory_segments.append(seg)
#     except rpt.exceptions.BadSegmentationParameters:
#         pass
#     return trajectory_segments

def find_trajectory_segments(
    surprises: np.ndarray,
    rewards: np.ndarray,
    lambda_returns: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    penalty=None,  # Now accepts None or a manually set penalty
    trajectory_id=None,
) -> List[Dict]:
    """
    Segment a trajectory using PELT/RBF.
    
    If penalty is None (default), it is calculated using a BIC-like heuristic 
    to statistically determine the optimal number of segments (K).
    """
    
    # 1. Feature Preparation (Same as original)
    # Features: [lambda_returns, surprises, rewards]
    features = np.column_stack((lambda_returns, surprises, rewards))
    trajectory_len = features.shape[0]
    feature_dim = features.shape[1] # 3 dimensions
    
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # --- Hyperparameters ---
    min_size = max(3, trajectory_len // 20)
    gamma_val = 2 # TUNE: Kernel sensitivity
    
    # 2. Penalty Selection (Crucial Change)
    if penalty is None:
        # Use a simplified, linear penalty factor. 
        # The factor is chosen based on the dimension of the feature space.
        # We keep the factor low to encourage splitting.
        
        penalty_factor = 1.5*feature_dim 
        
        # Use a simpler, non-log penalty. If 6 is too low (over-segments), 
        penalty = penalty_factor * np.log(trajectory_len)
        
        # Check if the trajectory is extremely long; if so, apply log scaling
        if trajectory_len > 1000:
            penalty = penalty_factor * np.log(trajectory_len)
        
    print(f"No penalty provided. Calculated penalty: {penalty:.2f}")
    
    # 3. KernelCPD/PELT Setup (Robust Penalty Search)
    try:
        # Define the Cost Model (RBF)
        cost_model = costs.CostRbf(gamma=gamma_val).fit(features) 

        # Use PELT (the optimal algorithm for penalized search)
        algo = rpt.Pelt(custom_cost=cost_model, min_size=min_size) 
        algo.fit(features)

        # 4. Predict Breakpoints using the statistically-derived penalty
        break_points = algo.predict(pen=penalty)
        
        # 5. Finalize breakpoints
        break_points = [0] + break_points
        
    except rpt.exceptions.BadSegmentationParameters:
        # Fallback: treat as a single segment
        break_points = [0, trajectory_len]

    # 6. Process Segments (Same as original)
    trajectory_segments = []
    for start, end in zip(break_points[:-1], break_points[1:]):
        if start >= end:
            continue
            
        seg = {
            "start": start,
            "end": end,
            "features": tuple(get_segment_features(start, end, surprises, rewards, lambda_returns, actions, states, trajectory_len).values()),
            "surprises": surprises[start:end],
            "return": np.mean(rewards[start:end]),
        }
        if trajectory_id is not None:
            seg["trajectory_id"] = trajectory_id
        trajectory_segments.append(seg)
        
    return trajectory_segments

def get_segment_features(seg_start:int, seg_end:int ,surprises:np.ndarray,rewards:np.ndarray, lambda_returns:np.ndarray, actions:np.ndarray, states:np.ndarray, trajectory_len:int):
    """
    Computes the features for a segment.
    """
    feature_names = ["λ_ret", "λ_ret_std", "surprise", "surprise_std", "reward", "reward_std", "length", "pos_start", "pos_end", "action_diversity", "state_diversity"]
    features = {}
    features["λ_ret"] = np.mean(lambda_returns[seg_start:seg_end])
    #features["λ_ret_std"] = np.std(elegibility_traces[seg_start:seg_end])
    features["surprise"] = np.mean(surprises[seg_start:seg_end])
    features["surprise_std"] = np.std(surprises[seg_start:seg_end])
    features["reward"] = np.mean(rewards[seg_start:seg_end])
    features["reward_std"] = np.std(rewards[seg_start:seg_end])
    features["length"] = (seg_end - seg_start)
    features["pos_start"] = seg_start
    features["pos_end"] = seg_end
    features["action_diversity"] = len(set(actions[seg_start:seg_end])) / len(actions[seg_start:seg_end]) if len(actions[seg_start:seg_end]) > 0 else 0.0
    features["state_diversity"] = len(set(states[seg_start:seg_end])) / len(states[seg_start:seg_end]) if len(states[seg_start:seg_end]) > 0 else 0.0
    return features
 
def cluster_segments(
    segments,
    include_features=None,
    eps=1.5,
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

    def find_knee_point_heuristic(X_scaled: np.ndarray, k: int = 12) -> float:
        """
        Numerically estimates the optimal DBSCAN epsilon (eps) value 
        by finding the 'knee' in the k-distance graph using a maximum distance heuristic.

        Args:
            X_scaled: The standardized feature array of segments.
            k: The MinPts value (the k for k-distance).

        Returns:
            The distance value at the detected knee point.
        """
        N = X_scaled.shape[0]
        if N <= k:
            print("Warning: Not enough samples for k-distance calculation. Using default eps=1.0.")
            return 1.0

        # 1. Calculate k-distances
        # k-th neighbor distance is at index k-1 in the returned array
        nn = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)

        # Extract the k-distance (distance to the k-th nearest neighbor) and sort it
        k_distance_sorted = np.sort(distances[:, k - 1])
        
        # 2. Define the baseline for the heuristic
        # The baseline is the line connecting the first point (0, y_0) and the last point (N-1, y_N-1)
        x = np.arange(N)
        
        # Line defined by y = m*x + c
        # (x_0, y_0) is (0, k_distance_sorted[0])
        # (x_N-1, y_N-1) is (N-1, k_distance_sorted[N-1])
        
        # Slope (m)
        m = (k_distance_sorted[N - 1] - k_distance_sorted[0]) / (N - 1)
        
        # Y-intercept (c)
        c = k_distance_sorted[0]
        
        # 3. Calculate the distance from each point to the baseline line (Perpendicular Distance)
        # The line equation is implicit: Ax + By + C = 0
        # Here: m*x - 1*y + c = 0. So, A=m, B=-1, C=c.
        
        # Perpendicular distance formula: |A*x_i + B*y_i + C| / sqrt(A^2 + B^2)
        distances_to_line = np.abs(m * x - k_distance_sorted + c) / np.sqrt(m**2 + 1)
        
        # 4. Find the index corresponding to the maximum distance
        knee_index = np.argmax(distances_to_line)
        
        # 5. The optimal epsilon is the k-distance value at the knee index
        optimal_eps = k_distance_sorted[knee_index]
    
        return optimal_eps

    estimated_eps = find_knee_point_heuristic(X, k=2*min_samples)
    estimated_eps = max(estimated_eps, 1e-6)  # Ensure eps is not too small
    print(f"Estimated DBSCAN eps: {estimated_eps:.4f} (using min_samples={min_samples})")
    clustering = DBSCAN(eps=estimated_eps, min_samples=min_samples).fit(X)
    
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

def js_divergence_per_state(policy_p: EmpiricalPolicy, policy_q: EmpiricalPolicy, all_actions:set,alpha: float = 0.1, epsilon: float = 1e-12):
    # 1. Determine the union of states for comparison
    states = set(policy_p._state_action_map.keys()) | set(policy_q._state_action_map.keys())
    if not states:
        return {}, 0.0

    js_per_state = {}

    for state in states:
        # Get actions observed for the current state in either policy P or Q.
        # This is the correct local action space for the divergence calculation at state s.
        p_probs = np.array([policy_p.get_action_probability(state, a, alpha) for a in all_actions])
        q_probs = np.array([policy_q.get_action_probability(state, a, alpha) for a in all_actions])

        # Calculate the average distribution M(a|s)
        m_probs = 0.5 * (p_probs + q_probs)

        # --- Stable KL Calculation: D_KL(P || M) ---
        # Only perform the calculation where P is non-zero
        non_zero_p_indices = p_probs > epsilon
        
        kl_pm = np.sum(
            p_probs[non_zero_p_indices] * np.log(p_probs[non_zero_p_indices] / m_probs[non_zero_p_indices])
        )

        # --- Stable KL Calculation: D_KL(Q || M) ---
        # Only perform the calculation where Q is non-zero
        non_zero_q_indices = q_probs > epsilon
        
        kl_qm = np.sum(
            q_probs[non_zero_q_indices] * np.log(q_probs[non_zero_q_indices] / m_probs[non_zero_q_indices])
        )

        # --- Final JS Divergence ---
        js_per_state[state] = 0.5 * (kl_pm + kl_qm)

    mean_js = float(np.mean(list(js_per_state.values())))
    return js_per_state, mean_js

def graph_from_policy(policy:EmpiricalPolicy)->nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for (state, action, next_state), count in policy._edge_count.items():
        if next_state is None:
            continue
        if not G.has_node(state):
            G.add_node(state)
        if not G.has_node(next_state):
            G.add_node(next_state)
        if G.has_edge(state, next_state):
            G[state][next_state]['weight'] += count
            G[state][next_state]['actions'].add(action)
        else:
            G.add_edge(state, next_state, weight=count, actions={action})
    return G

def policy_comparison(curr_policy:EmpiricalPolicy, prev_policy:EmpiricalPolicy, all_actions:set)->dict:
    """
    Compare two policies based on their trajectory statistics.
    """
    per_state_js_div, mean_js_div = js_divergence_per_state(curr_policy, prev_policy, all_actions)

    metrics = {
        "node_overlap": len(set(curr_policy.states) & set(prev_policy.states))/max(len(curr_policy.states | prev_policy.states), 1),
        "edge_overlap": len(set(curr_policy.edges) & set(prev_policy.edges))/max(len(curr_policy.edges | prev_policy.edges), 1),
        "js_divergence": mean_js_div,
        "added_nodes": len(set(curr_policy.states) - set(prev_policy.states)),
        "removed_nodes": len(set(prev_policy.states) - set(curr_policy.states)),
        "added_edges": len(set(curr_policy.edges) - set(prev_policy.edges)),
        "removed_edges": len(set(prev_policy.edges) - set(curr_policy.edges)),
        "action_agreeement": sum(1 for state in curr_policy.states if 
                                np.argmax([curr_policy.get_action_probability(state, a) for a in all_actions]) == 
                                np.argmax([prev_policy.get_action_probability(state, a) for a in all_actions]))/max(curr_policy.num_states, 1),
    }
    return metrics, per_state_js_div


def get_trajectory_action_ngrams(trajectory:Trajectory, n:int)->list:
    """
    Get n-grams of states and actions from a trajectory.
    """
    ngrams = []
    for i in range(len(trajectory) - n + 1):
        action_ngram = tuple(transition.action for transition in trajectory[i:i+n])
        ngrams.append(action_ngram)
    return ngrams


def get_steps_for_state(trajectories:Iterable, state:any)->list:
    """
    Get the list of step indices where the given state occurs in the trajectory.
    """
    steps = []
    for trajectory in trajectories:
        for i, transition in enumerate(trajectory.transitions):
            if transition.state == state:
                steps.append(i)
    return steps




#### NEW from experiments/generalization_experiment.py ####

### Emprical Policy Builders ###

def load_trajectories(json_file, max_trajectories=None, load_metadata=True):
    """
    Load trajectories from a JSON file.
    Args:
        json_file (str): Path to the JSON file.
        max_trajectories (int, optional): Maximum number of trajectories to load. If None, load all.
        load_metadata (bool): Whether to load metadata from the JSON file.
    Returns:
        list: List of loaded trajectories.
        dict: Metadata dictionary if load_metadata is True, else {}.
    """
    trajectories, metadata = load_trajectories_from_json(json_file, load_metadata=load_metadata, max_trajectories=max_trajectories, 
                                                         action_encoder=aidojo_action_type_from_dict,
                                                         state_encoder=numpy_default)
    print(f"Loaded {len(trajectories)} trajectories from {json_file}")
    return trajectories, metadata

def build_empirical_policy_from_list(trajectories:list, max_trajectories, action_space=None)-> tuple[EmpiricalPolicy, list]:
    """
    Builds an empirical policy from a list of trajectories.
    Args:
        trajectories (list): List of Trajectory objects.
        max_trajectories (int): Maximum number of trajectories to load.
        action_space (Iterable): Optional explicit action space.
    Returns:
        EmpiricalPolicy: The constructed empirical policy.
        list: List of loaded trajectories.
    """
    if max_trajectories:
        trajectories = trajectories[:max_trajectories]
    empirical_policy = EmpiricalPolicy(trajectories, action_space=action_space)
    return empirical_policy, trajectories

def build_empirical_policy_from_file(path, max_trajectories:int, action_space=None)-> tuple[EmpiricalPolicy, list[Trajectory]]:
    """
    Builds an empirical policy from trajectory data stored in a JSON file.
    Args:
        path (str): Path to the JSON file containing trajectory data.
        max_trajectories (int): Maximum number of trajectories to load.
        action_space (Iterable): Optional explicit action space.
    Returns:
        dict: Dictionary with empirical policies {pre_adapt_policy, post_adapt_policy}
        dict: Dictionary with loaded trajectories {pre_adapt_trajectories, post_adapt_trajectories}
    """
    # load the trajectories from file
    print(f"[Trajectory processing & EP build] {path}")
    trajectories, _ = load_trajectories(path, max_trajectories=max_trajectories, load_metadata=False)
    empirical_policy = EmpiricalPolicy(trajectories, action_space=action_space)
    return empirical_policy, trajectories

def split_trajectories_into_policies(trajectories: Iterable[Trajectory], action_space=None, test_ratio: float = 0.5) -> tuple[EmpiricalPolicy, EmpiricalPolicy]:
    """
    Randomly splits a list of trajectories and generates two EmpiricalPolicies.

    Args:
        trajectories (Iterable[Trajectory]): The list of trajectories to split.
        action_space (Iterable, optional): Explicit action space for the policies.
        test_ratio (float, optional): The ratio of trajectories to include in the second policy (test set). 
                                      Defaults to 0.5 (even split). Should be between 0.0 and 1.0.

    Returns:
        tuple[EmpiricalPolicy, EmpiricalPolicy]: Two empirical policies created from the split (train, test).
    """
    # Convert to list if it's not already, to allow shuffling
    traj_list = list(trajectories)
    # Ensure that we have enough trajectories to split
    if len(traj_list) < 2:
        raise ValueError("Not enough trajectories to split.")
    # Ensure that the test ratio is valid
    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError("test_ratio must be between 0.0 and 1.0.")

    # Shuffle in place
    random.shuffle(traj_list)
    
    # Calculate split index
    total_count = len(traj_list)
    test_count = int(total_count * test_ratio)
    split_idx = total_count - test_count
    
    # Split
    train_trajectories = traj_list[:split_idx]
    test_trajectories = traj_list[split_idx:]
    
    # Create policies
    train_policy = EmpiricalPolicy(train_trajectories, action_space=action_space)
    test_policy = EmpiricalPolicy(test_trajectories, action_space=action_space)
    
    return train_policy, test_policy

### Trajectory Distance Metrics ###

def get_transition_probabilities(policy: EmpiricalPolicy):
    """
    Extracts P(next_state | state) for the entire graph based on empirical counts.
    Args:
        policy (EmpiricalPolicy): The empirical policy containing edge counts.
    Returns: dict[state] -> list of (next_state, probability)
    """
    transitions = collections.defaultdict(lambda: collections.defaultdict(int))
    state_totals = collections.defaultdict(int)

    # Aggregate counts from the edge_count map
    # structure: (state, action, next_state) -> count
    for (s, a, next_s), count in policy._edge_count.items():
        transitions[s][next_s] += count
        state_totals[s] += count

    # Normalize to probabilities
    prob_map = {}
    for s, next_states_dict in transitions.items():
        total = state_totals[s]
        # Create list of (next_state, prob)
        prob_map[s] = [(ns, count / total) for ns, count in next_states_dict.items()]
    return prob_map

def compute_tvd(dist1, dist2, action_set):
    """
    Calculate the Total Variation Distance (TVD) between two action distributions.
    Args:
        dist1 (dict): The first action distribution.
        dist2 (dict): The second action distribution.
        action_set (set): The set of all possible actions.
    Returns:
        float: The TVD between the two policies in the given state.
    """

    tvd = 0.0
    for action in action_set:
        p1 = dist1.get(action, 0.0)
        p2 = dist2.get(action, 0.0)
        tvd += abs(p1 - p2)
    tvd *= 0.5
    return tvd

def compute_js_distance(dist1, dist2, action_set):
    """
    Computes Jensen-Shannon DISTANCE (sqrt(JSD)) for Policy Alignment.
    """
    # 1. Ensure consistent order of actions (Critical!)
    # Convert dicts to arrays, filling missing actions with 0.0
    actions = sorted(list(action_set), key=lambda x: str(x))  # Sort by string representation for consistency
    p1 = np.array([dist1.get(a, 0.0) for a in actions])
    p2 = np.array([dist2.get(a, 0.0) for a in actions])

    # 2. Compute Metric (Returns sqrt(JSD))
    # 'base=2' ensures the range is [0, 1]
    return jensenshannon(p1, p2, base=2)

### Bisimulation Metric Variants ###
def bisimulation_metric_relaxed(cost, T1, T2, gamma=0.95, eps=1e-7, max_iter=200, SAFE_MAX=1e8, verbose=False):
    """
    Optimized Relaxed (Hausdorff) Bisimulation Metric using vectorized NumPy operations.
    Explicitly penalizes structural mismatches (one terminal, one non-terminal).
    Args:
        cost: (N1 x N2) Initial cost matrix (e.g., reward difference or TVD).
        T1: (N1 x N1) Transition matrix for Policy 1.
        T2: (N2 x N2) Transition matrix for Policy 2.
        gamma: Discount factor.
        eps: Convergence threshold.
        max_iter: Maximum iterations.
        SAFE_MAX: Large constant to represent "infinite" distance.
        verbose: Whether to print iteration logs.
    """
    # prepare variables
    N1, N2 = cost.shape
    d = cost.astype(np.float64).copy()

    has_child1 = np.any(T1 > 0, axis=1)
    has_child2 = np.any(T2 > 0, axis=1)

    # Masks
    # both have children
    active_mask = has_child1[:, None] & has_child2[None, :]
    # one is terminal, one is not
    mismatch_mask = has_child1[:, None] ^ has_child2[None, :]
    # both are terminal
    terminal_mask = ~has_child1[:, None] & ~has_child2[None, :]

    # Precompute children indices
    children1 = [np.where(T1[u] > 0)[0] for u in range(N1)]
    children2 = [np.where(T2[v] > 0)[0] for v in range(N2)]

    for it in range(max_iter):
        d_prev = d.copy()

        # --- Vectorized Hausdorff update ---
        # Upddate: d(u, v) = c(u, v) + gamma * max( E_u[min_v d], E_v[min_u d] )
        # Forward direction (min over children of v)
        M1 = np.full((N1, N2), SAFE_MAX, dtype=np.float64)
        for v, kids_v in enumerate(children2):
            if len(kids_v) > 0:
                # Broadcasting: take min across the children's axis
                M1[:, v] = np.min(d[:, kids_v], axis=1)
        future_uv = T1 @ M1

        # Backward direction (min over children of u)
        M2 = np.full((N1, N2), SAFE_MAX, dtype=np.float64)
        for u, kids_u in enumerate(children1):
            if len(kids_u) > 0:
                M2[u, :] = np.min(d[kids_u, :], axis=0)
        future_vu = M2 @ T2.T

        discrepancy = np.maximum(future_uv, future_vu)

        # --- Apply updates ---
        d[active_mask] = cost[active_mask] + gamma * discrepancy[active_mask]
        d[mismatch_mask] = SAFE_MAX
        d[terminal_mask] = cost[terminal_mask]

        # Convergence check
        delta = np.max(np.abs(d - d_prev))
        if verbose:
            print(f"[Relaxed Optimized] iter {it}: delta={delta:.6e}")
        if delta < eps:
            break
    return d

### Empirical Policy Similarity ###
def find_psm_mapping(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions,
                     gamma=0.95, iterations=3, normalize_cost_matrix=False, REWARD_SCALE=100.0):
    """
    Find cost of matching states between two empirical policies using the policy similarity metric (PSM)[https://arxiv.org/pdf/2101.05265].
    Args:
        policy1 (EmpiricalPolicy): The first empirical policy.
        policy2 (EmpiricalPolicy): The second empirical policy.
        global_actions (set): The set of all possible actions (common across policies).
        gamma (float): Discount factor for bisimulation metric.
        iterations (int): Number of iterations for bisimulation refinement.
        normalize_cost_matrix (bool): Whether to normalize the final cost matrix.
        REWARD_SCALE (float): Scaling factor for reward differences.
    Returns:
        cost_matrix (np.ndarray): The final cost matrix between states of the two policies.
        row_ind (np.ndarray): Row indices of the optimal matching.
        col_ind (np.ndarray): Column indices of the optimal matching.
        nodes1 (list): List of states in policy1.
        nodes2 (list): List of states in policy2.
        n1_idx (dict): Mapping from state to index in policy1.
        n2_idx (dict): Mapping from state to index in policy2.
        d1_map (dict): Action distribution map for policy1 states.
        d2_map (dict): Action distribution map for policy2 states.
    """
    # Get nodes from both policies
    nodes1 = list(policy1.states)
    nodes2 = list(policy2.states)
    n1_len = len(nodes1)
    n2_len = len(nodes2)
    n1_idx = {n: i for i, n in enumerate(nodes1)}
    n2_idx = {n: i for i, n in enumerate(nodes2)}
  
    # Pre-compute action distributions for all states in both policies with smoothing (alpha=0.001)
    d1_map = {n: policy1.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes1}
    d2_map = {n: policy2.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes2}

    # 3. Initialize local cost matrix
    # Each entry cost_matrix[i,j] represents the cost of matching state nodes1[i] with nodes2[j]
    # Initialize with reward differences and terminal state handling
    # for terminal states, use average value difference; for non-terminal, use policy distribution distance (js distance)
    cost_matrix = np.zeros((n1_len, n2_len), dtype=np.float64)
    for i, u in enumerate(nodes1):
        for j, v in enumerate(nodes2):
            is_terminal_1 = len(policy1._state_action_map.get(u, {})) == 0
            is_terminal_2 = len(policy2._state_action_map.get(v, {})) == 0

            if is_terminal_1 and is_terminal_2:
                r1 = policy1.get_average_value(u)
                r2 = policy2.get_average_value(v)
                cost_matrix[i, j] = np.tanh(abs(r1 - r2) / REWARD_SCALE)
            elif is_terminal_1 or is_terminal_2:
                cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, j] = compute_tvd(d1_map[u], d2_map[v], global_actions)
                #cost_matrix[i, j] = compute_js_distance(d1_map[u], d2_map[v], global_actions)

    # 4. Convert empirical policies to transition matrices
    def policy_to_matrix(policy, nodes, node_to_idx):
        n = len(nodes)
        T = np.zeros((n, n), dtype=np.float64)
        for i, u in enumerate(nodes):
            transitions = policy.get_target_transitions(u, normalize=True)
            for child, prob in transitions:
                c_idx = node_to_idx[child] if child in node_to_idx else child
                T[i, c_idx] = prob
        return T

    T1 = policy_to_matrix(policy1, nodes1, n1_idx)
    T2 = policy_to_matrix(policy2, nodes2, n2_idx)


    cost_matrix = bisimulation_metric_relaxed(
        cost=cost_matrix,
        T1=T1,
        T2=T2,
        gamma=gamma,
        eps=1e-7,
        max_iter=iterations,
        verbose=True
    )
    
    # Normalize (optional)
    if normalize_cost_matrix:
        scaling_factor = 1.0 / (1.0 - gamma)
        cost_matrix = cost_matrix / scaling_factor

    # 6. Hungarian Algorithm for optimal node matching
    print("Finding Optimal Node Matching using Hungarian Algorithm...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(f"Matched {len(row_ind)} states.")

    return cost_matrix, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map    


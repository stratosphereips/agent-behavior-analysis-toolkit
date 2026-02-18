import numpy as np
from typing import Iterable, Dict, Any, List
import scipy.special
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from trajectory import EmpiricalPolicy, Trajectory, Transition
import networkx as nx

def laplace_smoothing(counts: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply Laplace smoothing to counts and return normalized probabilities.
    """
    smoothed = counts + alpha
    return smoothed / smoothed.sum()

def state_kl_divergence(counts1: Dict, counts2: Dict, global_keys: Iterable, alpha=1.0) -> float:
    """
    Compute the KL divergence between two distributions of key frequencies.
    Args:
        counts1: counts for the first distribution
        counts2: counts for the second distribution
        global_keys: global key space   
        alpha: smoothing parameter
    Returns:
        kl_div: KL divergence between the two distributions
    """
    vec_counts1 = np.array([counts1.get(key, 0) for key in global_keys])
    vec_counts2 = np.array([counts2.get(key, 0) for key in global_keys])
    p_probs = laplace_smoothing(vec_counts1, alpha)
    q_probs = laplace_smoothing(vec_counts2, alpha)
    kl_div_per_key = scipy.special.kl_div(p_probs, q_probs)
    return np.sum(kl_div_per_key)

def topological_shift(current_state_visitation: Dict, previous_state_visitation: Dict, noise_value: float=0.0) -> float:
    """
    Compute the topological shift between two policies represented as JS divergence between their state visitation distributions.
    The noise_value is subtracted from the JS divergence to account for noise in the state visitation distributions.
    Args:
        current_state_visitation: state visitation distribution of the current policy
        previous_state_visitation: state visitation distribution of the previous policy
        noise_value: noise value to subtract from the JS divergence

    Returns:
        topo_shift: topological shift between the two policies
    """
    all_states = set(current_state_visitation.keys()).union(set(previous_state_visitation.keys()))
    js_div = state_js_divergence(current_state_visitation, previous_state_visitation, all_states)
    topo_shift = max(0, js_div-noise_value)    
    return topo_shift

def state_js_divergence(counts1: Dict, counts2: Dict, global_keys: Iterable, alpha=1.0) -> float:
    """
    Robust JS Divergence using Scipy's implementation (Base 2).
    Returns value in [0, 1].
    """
    # 1. Align vectors
    keys = list(global_keys)
    vec1 = np.array([counts1.get(k, 0) for k in keys], dtype=float)
    vec2 = np.array([counts2.get(k, 0) for k in keys], dtype=float)

    # 2. Add Laplace Smoothing (Alpha) to raw counts
    vec1 += alpha
    vec2 += alpha

    # 3. Normalize to probabilities
    p = vec1 / np.sum(vec1)
    q = vec2 / np.sum(vec2)

    # 4. Compute JSD (Base 2 ensures bound [0, 1])
    return jensenshannon(p, q, base=2)**2  # Square it because scipy returns Distance (sqrt(div))

def strategic_shift(current_policy, previous_policy, global_actions, noise_value=0.0):
    """
    Computes Strategic Shift using Weighted JSD on shared states.
    Bounded [0, 1].
    """
    # 1. Identify Shared States
    s_curr = set(current_policy._state_visitation_count.keys())
    s_prev = set(previous_policy._state_visitation_count.keys())
    shared_states = s_curr.intersection(s_prev)

    if not shared_states:
        return 1.0  # Max divergence if no overlap

    # 2. Calculate Weights (Re-normalized to sum to 1.0 over shared set)
    # We average the occupancy from both policies to be symmetric
    w_num = []
    jsd_vals = []
    
    for s in shared_states:
        # Get raw counts for this state
        c_curr = current_policy._state_visitation_count[s]
        c_prev = previous_policy._state_visitation_count.get(s, 0) # Should exist if intersection
        
        # Average Weight: (P_curr(s) + P_prev(s)) / 2
        # Note: We use raw counts here as proxy for importance, then normalize later
        weight = (c_curr + c_prev) / 2.0
        w_num.append(weight)

        # Get Action Distributions (assuming helper returns raw counts dict)
        act_counts_1 = current_policy._state_action_map[s]
        act_counts_2 = previous_policy._state_action_map[s]
        
        # Compute JSD for this state's policy
        # Reuse state_js_divergence logic but for actions
        val = state_js_divergence(act_counts_1, act_counts_2, global_actions, alpha=1.0)
        jsd_vals.append(val)

    # 3. Compute Weighted Average
    w_num = np.array(w_num)
    if np.sum(w_num) == 0: return 0.0
    
    weights = w_num / np.sum(w_num) # Sums to 1.0
    
    raw_strat_shift = np.sum(weights * np.array(jsd_vals))
    
    # 4. Apply Noise Threshold
    return max(0.0, raw_strat_shift - noise_value)

def traversal_depth(trajectories: List[Trajectory]) -> float:
    """
    Computes the traversal depth of a policy given a set of trajectories.
    Args:
        trajectories: List of trajectories
    Returns:
        traversal_depth: Traversal depth of the policy
    """
    if len(trajectories) == 0:
        return 0.0
    
    # compute state visitation distribution of the trajectories
    graph = nx.DiGraph()
    start_nodes = set()
    for trajectory in trajectories:
        start_nodes.add(trajectory.states[0])
        for transition in trajectory.transitions:
            graph.add_edge(transition.state, transition.next_state, weight=1.0)
    root_id = "VIRTUAL_ROOT"
    graph.add_node(root_id)
    for start_node in start_nodes:
        graph.add_edge(root_id, start_node, weight=0)
    try:
        lengths = nx.single_source_dijkstra_path_length(graph, root_id, weight='weight')
        # Exclude the root (dist=0)
        depths = [d for n, d in lengths.items() if n != root_id]
        
        # Return max depth
        return max(depths) if depths else 0.0

    except nx.NetworkXNoPath:
        return 0.0 
    
def compute_entropy_metrics(state_counts, action_counts, action_space_size):
    """
    Computes Spatial Focus (H_S) and Strategic Confidence (H_pi) from count dictionaries.

    Args:
        state_counts: Dict {state_id: count}
        action_counts: Dict {state_id: {action_id: count}}
        action_space_size: Int (Total number of possible actions, e.g., 2 or 4)

    Returns:
        H_S (float): Spatial Entropy (Spread)
        H_pi (float): Weighted Action Entropy (Uncertainty)
    """
    total_visits = sum(state_counts.values())
    if total_visits == 0:
        return 0.0, 0.0

    # --- Metric 1: Spatial Focus (H_S) ---
    # P(s) = count(s) / total_steps
    p_s_values = np.array(list(state_counts.values())) / total_visits
    h_s_bits = entropy(p_s_values, base=2)
    effective_state_coverage = 2 ** h_s_bits
    # --- Metric 2: Strategic Confidence (H_pi) ---
    weighted_h_pi = 0.0

    for state, count in state_counts.items():
        p_s = count / total_visits
        
        # Get action counts for this specific state (default to empty if missing)
        s_actions = action_counts.get(state, {})
        
        # Build vector [count_a0, count_a1, ...]
        # We assume action_ids are integers 0..N-1
        counts_vec = np.zeros(action_space_size)
        for act_id, act_count in s_actions.items():
            if 0 <= act_id < action_space_size:
                counts_vec[act_id] = act_count
        
        # Laplace Smoothing (Alpha=1)
        # Prevents "0 probability" errors for unvisited actions
        counts_vec += 1.0 
        
        # Normalize to get Policy distribution \pi(a|s)
        pi_s = counts_vec / np.sum(counts_vec)
        
        # Calculate entropy of the policy at this state
        h_pi_s = entropy(pi_s, base=2)
        
        # Add to weighted sum
        weighted_h_pi += p_s * h_pi_s
    
    # Normalize H_pi to [0, 1]
    max_entropy = np.log2(action_space_size)
    if max_entropy == 0:
        empirical_policy_certainty = 1.0
    else:
        normalized_h_pi = weighted_h_pi / max_entropy
        empirical_policy_certainty = 1.0 - normalized_h_pi

    return effective_state_coverage, empirical_policy_certainty


def calculate_temporal_action_entropy(trajectories, num_actions=4):
    """
    Computes the Survival-Weighted Temporal Action Entropy (H_tau).
    
    Args:
        trajectories: List of lists, where each inner list is a sequence of action indices.
                      e.g., [[0, 1, 1], [0, 2], [0, 1, 1, 2]]
        num_actions:  Integer, total number of possible actions in the env (e.g., 4 for FrozenLake).
        
    Returns:
        float: A scalar between 0.0 (Deterministic) and 1.0 (Maximum Entropy).
    """
    # 1. Handle empty input
    if not trajectories:
        return 0.0
        
    # 2. Determine maximum trajectory length
    max_len = max(len(t) for t in trajectories)
    
    # 3. Initialize counts: [time_step, action_index]
    # This matrix counts how many agents took action 'a' at step 't'
    action_counts = np.zeros((max_len, num_actions))
    
    # 4. Fill the matrix
    for traj in trajectories:
        for t, transition in enumerate(traj):
            if 0 <= transition.action < num_actions: # Safety check
                action_counts[t, transition.action] += 1
                
    # 5. Compute Entropy per time step
    step_entropies = []
    step_weights = []
    
    for t in range(max_len):
        # Get the distribution of actions at step t
        counts = action_counts[t]
        total_survivors = np.sum(counts)
        
        # Only compute if there is at least 1 survivor
        if total_survivors > 0:
            # Scipy's entropy function automatically normalizes counts to probabilities.
            # We use base=num_actions so the result is normalized to [0, 1].
            # 0 = All survivors took the same action.
            # 1 = Survivors split evenly among all actions.
            H_t = entropy(counts, base=num_actions)
            
            step_entropies.append(H_t)
            step_weights.append(total_survivors)
            
    # 6. Compute Weighted Average
    # We weight each step's entropy by the number of survivors at that step.
    # This prevents the tail end of long, rare trajectories from dominating the metric.
    total_steps = sum(step_weights)
    
    if total_steps == 0:
        return 0.0
        
    weighted_H_tau = np.average(step_entropies, weights=step_weights)
    
    return weighted_H_tau
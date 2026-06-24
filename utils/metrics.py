import numpy as np
from typing import Iterable, Dict, Any, List
import scipy.special
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from trajectory import EmpiricalPolicy, Trajectory, Transition
import networkx as nx
from collections import Counter
import ot

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

def compute_decomposed_jsd(counts_A:dict, counts_B:dict):
    """
    Computes the decomposed JSD using vectorized operations.
    
    Args:
        counts_A (np.ndarray): 1D array of raw visit counts for Agent A.
        counts_B (np.ndarray): 1D array of raw visit counts for Agent B.
        
    Returns:
        dict: A breakdown of the JSD components (in bits).
    """
    # 1. Create a unified set of all keys (states)
    all_keys = sorted(list(set(counts_A.keys()) | set(counts_B.keys())), key=str)
    
    # 2. Build aligned arrays
    # We use a list comprehension or a loop to ensure indices match the global key set
    array_A = np.array([counts_A.get(k, 0) for k in all_keys], dtype=float)
    array_B = np.array([counts_B.get(k, 0) for k in all_keys], dtype=float)

    # 3. Normalize to probabilities
    sum_A = np.sum(array_A)
    sum_B = np.sum(array_B)
    
    if sum_A == 0 or sum_B == 0:
        raise ValueError("Cannot compute divergence: An agent has zero total visits.")
        
    A = array_A / sum_A
    B = array_B / sum_B
    
    # 2. Create Boolean Masks for the support regions
    mask_overlap = (A > 0) & (B > 0)
    mask_unique_A = (A > 0) & (B == 0)
    mask_unique_B = (A == 0) & (B > 0)
    
    # 3. Compute Overlapping JSD (Vectorized)
    # Slicing with the mask guarantees all values are > 0, 
    # so np.log2 will never throw a warning or NaN.
    A_ov = A[mask_overlap]
    B_ov = B[mask_overlap]
    M_ov = (A_ov + B_ov) / 2.0
    
    kl_a = A_ov * np.log2(A_ov / M_ov)
    kl_b = B_ov * np.log2(B_ov / M_ov)
    
    jsd_overlap = 0.5 * np.sum(kl_a) + 0.5 * np.sum(kl_b)
    
    # 4. Compute Non-Overlapping JSD strictly via probability mass
    p_A_unique = np.sum(A[mask_unique_A])
    p_B_unique = np.sum(B[mask_unique_B])
    
    # Because np.log2(2) is exactly 1.0, the constant simplifies to 0.5
    jsd_non_overlap = 0.5 * (p_A_unique + p_B_unique)
    
    # 5. Total JSD
    jsd_total = jsd_overlap + jsd_non_overlap
    
    return {
        "jsd_total": jsd_total,
        "jsd_overlap": jsd_overlap,
        "jsd_non_overlap": jsd_non_overlap,
        "p_A_unique": p_A_unique,
        "p_B_unique": p_B_unique
    }

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

def state_js_divergence(counts1: Dict, counts2: Dict, global_keys: Iterable) -> float:
    """
    Robust JS Divergence using Scipy's implementation (Base 2).
    Returns value in [0, 1].
    """
    # 1. Align vectors
    keys = list(global_keys)
    vec1 = np.array([counts1.get(k, 0) for k in keys], dtype=float)
    vec2 = np.array([counts2.get(k, 0) for k in keys], dtype=float)

    # 2. Normalize to probabilities
    p = vec1 / np.sum(vec1)
    q = vec2 / np.sum(vec2)

    # 3. Compute JSD (Base 2 ensures bound [0, 1])
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
        val = state_js_divergence(act_counts_1, act_counts_2, global_actions)
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
    unweighted_h_pi = 0.0
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
        unweighted_h_pi += h_pi_s
    
    # Normalize H_pi to [0, 1]
    max_entropy = np.log2(action_space_size)
    if max_entropy == 0:
        empirical_policy_certainty = 1.0
    else:
        normalized_h_pi = weighted_h_pi / max_entropy
        normalized_unweighted_h_pi = unweighted_h_pi / len(state_counts)
        empirical_policy_certainty = 1.0 - normalized_h_pi
        unweighted_policy_certainty = 1.0 - normalized_unweighted_h_pi  

    return effective_state_coverage, empirical_policy_certainty, unweighted_policy_certainty

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

def compute_ngram_histogram(trajectories, n=3, window_size=1)->dict:
    

    ngrams = []
    for traj in trajectories:
        for i in range(len(traj) - n + 1):
            ngram = tuple(transition.action for transition in traj[i:i+n])
            ngrams.append(ngram)
    ngram_counts = Counter(ngrams)
    return ngram_counts
    
def compute_ngram_jsd(trajectories1, trajectories2, n=3, window_size=1, alpha=1e-6, action_space_size=None)->float:
    ngram_counts1 = compute_ngram_histogram(trajectories1, n, window_size)
    ngram_counts2 = compute_ngram_histogram(trajectories2, n, window_size)
    all_ngrams = set(ngram_counts1.keys()) | set(ngram_counts2.keys())
    support_size = (action_space_size ** n) if action_space_size is not None else len(all_ngrams)
    vec1 = np.array([ngram_counts1.get(k, 0) for k in all_ngrams], dtype=float)
    vec2 = np.array([ngram_counts2.get(k, 0) for k in all_ngrams], dtype=float)
    total1 = np.sum(vec1)
    total2 = np.sum(vec2)
    p = (vec1 + alpha) / (total1 + alpha * support_size)
    q = (vec2 + alpha) / (total2 + alpha * support_size)
    return jensenshannon(p, q, base=2)

def compute_ngram_wasserstein_fast(trajectories1, trajectories2, global_ngrams, cost_matrix, n=3, window_size=1, alpha=1e-6) -> float:
    # 1. Extract empirical histograms
    ngram_counts1 = compute_ngram_histogram(trajectories1, n, window_size)
    ngram_counts2 = compute_ngram_histogram(trajectories2, n, window_size)
    
    # 2. Map directly to the canonical global support set
    vec1 = np.array([ngram_counts1.get(k, 0) for k in global_ngrams], dtype=float)
    vec2 = np.array([ngram_counts2.get(k, 0) for k in global_ngrams], dtype=float)
    
    # 3. Additive Smoothing and Normalization
    support_size = len(global_ngrams)
    total1 = np.sum(vec1)
    total2 = np.sum(vec2)
    
    p = (vec1 + alpha) / (total1 + alpha * support_size)
    q = (vec2 + alpha) / (total2 + alpha * support_size)
    
    # Enforce strict float precision for the OT solver
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 4. Compute optimal transport using the precomputed matrix
    wasserstein_dist = ot.emd2(p, q, cost_matrix)
    
    return float(wasserstein_dist)

def compute_perplexity_from_counts(counts:Dict[Any, float])->float:
    """
    Compute the perplexity of a probability distribution.
    Args:
        counts: counts for the distribution
    Returns:
        perplexity: perplexity of the distribution
    """
    vec_counts = np.array(list(counts.values()))
    entropy = scipy.stats.entropy(vec_counts, base=2)
    perplexity = 2 ** entropy
    return perplexity
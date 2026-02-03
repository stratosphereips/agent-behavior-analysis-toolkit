import numpy as np
from typing import Iterable, Dict
import scipy.special
from trajectory import EmpiricalPolicy, Trajectory, Transition

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

# def state_js_divergence(counts1: Dict, counts2: Dict, global_keys: Iterable, alpha=1.0) -> float:
#     """
#     Compute the JS divergence between two distributions of key frequencies.
#     Args:
#         counts1: counts for the first distribution
#         counts2: counts for the second distribution
#         global_keys: global key space
#         alpha: smoothing parameter
#     Returns:
#         js_div: JS divergence between the two distributions
#     """
#     vec_1 = np.array([counts1.get(key, 0) for key in global_keys])
#     vec_2 = np.array([counts2.get(key, 0) for key in global_keys])
    
#     p = laplace_smoothing(vec_1, alpha)
#     q = laplace_smoothing(vec_2, alpha)
#     m = 0.5 * (p + q)
    
#     kl_pm = np.sum(scipy.special.kl_div(p, m))
#     kl_qm = np.sum(scipy.special.kl_div(q, m))
    
#     return 0.5 * kl_pm + 0.5 * kl_qm

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

# def strategic_shift(current_policy: EmpiricalPolicy, previous_policy: EmpiricalPolicy, global_actions: Iterable, noise_value: float=0.0) -> float:
#     """
#     Compute the strategic shift between two policies represented as weighted KL divergence
#     between their action distributions in shared states. The noise_value is subtracted from the weighted KL divergence
#     to account for noise in the action distributions.
#     Args:
#         current_policy: current policy
#         previous_policy: previous policy
#         global_actions: global action space
#         noise_value: noise value to subtract from the weighted KL divergence
#     Returns:
#         strategic_shift: strategic shift between the two policies
#     """
#     shared_states = set(current_policy._state_visitation_count.keys()).intersection(set(previous_policy._state_visitation_count.keys()))
#     weighted_kl_div = 0
#     total_state_visitation = sum(current_policy._state_visitation_count.values())
#     state_visitation_prob = {state: current_policy._state_visitation_count[state]/total_state_visitation for state in shared_states}
#     for state in shared_states:
#         # compute the KL divergence between the two policies at the current state
#         state_kl_div = state_kl_divergence(current_policy._state_action_map[state], previous_policy._state_action_map[state], global_actions)
#         # weight the KL divergence by the state visitation frequency
#         weighted_kl_div += state_kl_div * state_visitation_prob[state]
#     strategic_shift = max(0, weighted_kl_div-noise_value)
#     return strategic_shift


from scipy.spatial.distance import jensenshannon

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
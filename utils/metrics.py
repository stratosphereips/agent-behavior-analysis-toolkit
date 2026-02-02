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

def state_kl_divergence(action_counts1: Dict, action_counts2: Dict, global_actions: Iterable, alpha=1.0) -> float:
    """
    Compute the KL divergence between two action distributions.
    Args:
        action_counts1: action counts for the first policy
        action_counts2: action counts for the second policy
        global_actions: global action space
        alpha: smoothing parameter
    Returns:
        kl_div: KL divergence between the two policies
    """
    vec_action_counts1 = np.array([action_counts1.get(action, 0) for action in global_actions])
    vec_action_counts2 = np.array([action_counts2.get(action, 0) for action in global_actions])
    p_probs = laplace_smoothing(vec_action_counts1, alpha)
    q_probs = laplace_smoothing(vec_action_counts2, alpha)
    kl_div_per_key = scipy.special.kl_div(p_probs, q_probs)
    return np.sum(kl_div_per_key)

def state_js_divergence(action_counts1: Dict, action_counts2: Dict, global_actions: Iterable, alpha=1.0) -> float:
    """
    Compute the JS divergence between two action distributions.
    Args:
        action_counts1: action counts for the first policy
        action_counts2: action counts for the second policy
        global_actions: global action space
        alpha: smoothing parameter
    Returns:
        js_div: JS divergence between the two policies
    """
    vec_1 = np.array([action_counts1.get(action, 0) for action in global_actions])
    vec_2 = np.array([action_counts2.get(action, 0) for action in global_actions])
    
    p = laplace_smoothing(vec_1, alpha)
    q = laplace_smoothing(vec_2, alpha)
    m = 0.5 * (p + q)
    
    kl_pm = np.sum(scipy.special.kl_div(p, m))
    kl_qm = np.sum(scipy.special.kl_div(q, m))
    
    return 0.5 * kl_pm + 0.5 * kl_qm

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

def strategic_shift(current_policy: EmpiricalPolicy, previous_policy: EmpiricalPolicy, global_actions: Iterable, noise_value: float=0.0) -> float:
    """
    Compute the strategic shift between two policies represented as weighted KL divergence
    between their action distributions in shared states. The noise_value is subtracted from the weighted KL divergence
    to account for noise in the action distributions.
    Args:
        current_policy: current policy
        previous_policy: previous policy
        global_actions: global action space
        noise_value: noise value to subtract from the weighted KL divergence
    Returns:
        strategic_shift: strategic shift between the two policies
    """
    shared_states = set(current_policy.get_state_counts().keys()).intersection(set(previous_policy.get_state_counts().keys()))
    weighted_kl_div = 0
    for state in shared_states:
        # compute the KL divergence between the two policies at the current state
        state_kl_div = state_kl_divergence(current_policy.get_state_counts()[state], previous_policy.get_state_counts()[state], global_actions)
        # weight the KL divergence by the state visitation frequency
        weighted_kl_div += state_kl_div * current_policy.get_state_counts()[state]
    strategic_shift = max(0, weighted_kl_div-noise_value)
    return strategic_shift
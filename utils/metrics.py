import numpy as np
from typing import Iterable, Dict, Any
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from collections import Counter
import ot

def compute_decomposed_jsd(counts_A:dict, counts_B:dict):
    """Computes the Jensen-Shannon divergence decomposed into overlap and non-overlap terms.

    Args:
        counts_A (dict): Raw visitation counts for agent A, keyed by state.
        counts_B (dict): Raw visitation counts for agent B, keyed by state.

    Returns:
        dict: Breakdown of the JSD (in bits) with keys ``jsd_total``,
        ``jsd_overlap``, ``jsd_non_overlap``, ``p_A_unique`` and
        ``p_B_unique``.

    Raises:
        ValueError: If either agent has zero total visits.
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

def state_js_divergence(counts1: Dict, counts2: Dict, global_keys: Iterable) -> float:
    """Computes the Jensen-Shannon divergence between two count distributions.

    Args:
        counts1 (Dict): Raw counts for the first distribution, keyed by state.
        counts2 (Dict): Raw counts for the second distribution, keyed by state.
        global_keys (Iterable): Key space to align both count dicts over.

    Returns:
        float: JS divergence (base 2) between the normalized distributions,
        bounded in [0, 1]. Returns 1.0 if either distribution has zero
        total mass.
    """
    # 1. Align vectors
    keys = list(global_keys)
    vec1 = np.array([counts1.get(k, 0) for k in keys], dtype=float)
    vec2 = np.array([counts2.get(k, 0) for k in keys], dtype=float)

    # 2. Normalize to probabilities
    s1, s2 = np.sum(vec1), np.sum(vec2)
    if s1 == 0 or s2 == 0:
        return 1.0  # Max divergence if either distribution is empty
    p = vec1 / s1
    q = vec2 / s2

    # 3. Compute JSD (Base 2 ensures bound [0, 1])
    return jensenshannon(p, q, base=2)**2  # Square it because scipy returns Distance (sqrt(div))

def strategic_shift(current_policy, previous_policy, global_actions, noise_value=0.0):
    """Computes strategic shift as the occupancy-weighted JSD of action distributions on shared states.

    Args:
        current_policy: Policy exposing ``_state_visitation_count`` and
            ``_state_action_map`` for the current period.
        previous_policy: Policy exposing the same attributes for the
            previous period.
        global_actions: Action space to align per-state action
            distributions over.
        noise_value (float): Value subtracted from the raw strategic shift
            to account for estimation noise. Defaults to 0.0.

    Returns:
        float: Occupancy-weighted average JS divergence between action
        distributions on states shared by both policies, bounded in
        [0, 1]. Returns NaN if the policies share no states.
    """
    # 1. Identify Shared States
    s_curr = set(current_policy._state_visitation_count.keys())
    s_prev = set(previous_policy._state_visitation_count.keys())
    shared_states = s_curr.intersection(s_prev)

    if not shared_states:
        return float('nan')  # Undefined: no shared states to compare action distributions on

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

def _stepwise_action_counts(trajectories, max_len, win_action="A_win", loss_action="A_lost") -> list:
    """Per-step raw action counts, padding episodes that end early.

    Once a trajectory terminates, it stops contributing to the raw per-step action
    counts used elsewhere (e.g. ``plot_action_per_step_distribution``), so the
    population being compared shrinks over time. Here we instead pad every step
    past a trajectory's end with a terminal pseudo-action -- ``win_action`` or
    ``loss_action``, chosen from its total reward -- so every step's distribution
    is taken over the same fixed population and reachability decay shows up
    directly as growing win/loss mass instead of vanishing support.

    Returns:
        list[Counter]: counts[t] is the action-count distribution at step t.
    """
    counts = [Counter() for _ in range(max_len)]
    for traj in trajectories:
        actions = traj.actions
        pad_token = win_action if traj.total_reward() > 0 else loss_action
        for t in range(max_len):
            action = actions[t] if t < len(actions) else pad_token
            counts[t][action] += 1
    return counts


def compute_stepwise_action_jsd(
    trajectories_a: Iterable,
    trajectories_b: Iterable,
    global_actions: Iterable,
    win_action="A_win",
    loss_action="A_lost",
) -> Dict[str, Any]:
    """Computes the per-step Jensen-Shannon divergence between two groups' action distributions.

    Trajectories that finish before the longest one in either group are padded
    with a terminal pseudo-action (``win_action``/``loss_action``, picked from
    the trajectory's total reward) so the divergence at every step is computed
    over the full population in both groups. See ``_stepwise_action_counts``.

    Args:
        trajectories_a: First group of trajectories (e.g. "seen").
        trajectories_b: Second group of trajectories (e.g. "unseen").
        global_actions: Canonical action space shared by both groups.
        win_action: Pad token used for steps after a winning trajectory ends.
        loss_action: Pad token used for steps after a losing trajectory ends.

    Returns:
        dict: ``steps`` (list[int]), ``jsd_per_step`` (list[float] in [0, 1],
        one per step), and ``mean_jsd`` (float, the mean of ``jsd_per_step``).
        Empty/NaN if both groups are empty.
    """
    trajectories_a = list(trajectories_a)
    trajectories_b = list(trajectories_b)
    max_len = max((len(t) for t in trajectories_a + trajectories_b), default=0)
    if max_len == 0:
        return {"steps": [], "jsd_per_step": [], "mean_jsd": float("nan")}

    global_keys = list(global_actions) + [win_action, loss_action]
    counts_a = _stepwise_action_counts(trajectories_a, max_len, win_action, loss_action)
    counts_b = _stepwise_action_counts(trajectories_b, max_len, win_action, loss_action)

    jsd_per_step = [
        float(state_js_divergence(counts_a[t], counts_b[t], global_keys))
        for t in range(max_len)
    ]
    return {
        "steps": list(range(max_len)),
        "jsd_per_step": jsd_per_step,
        "mean_jsd": float(np.mean(jsd_per_step)),
    }


def compute_ngram_histogram(trajectories, n=3, window_size=1)->dict:
    """Counts action n-grams observed across a set of trajectories.

    Args:
        trajectories: Iterable of trajectories, each a sequence of
            transitions exposing an ``action`` attribute.
        n (int): Length of the action n-grams to extract. Defaults to 3.
        window_size (int): Currently unused; reserved for future stride
            support. Defaults to 1.

    Returns:
        Counter: Counts of each action n-gram (tuple of length n) observed
        across all trajectories.
    """
    ngrams = []
    for traj in trajectories:
        for i in range(len(traj) - n + 1):
            ngram = tuple(transition.action for transition in traj[i:i+n])
            ngrams.append(ngram)
    ngram_counts = Counter(ngrams)
    return ngram_counts
    
def compute_ngram_wasserstein_from_counts(ngram_counts1, ngram_counts2, global_ngrams, cost_matrix, alpha=1e-6) -> float:
    """Computes the n-gram Wasserstein distance from precomputed n-gram counts.

    Same computation as ``compute_ngram_wasserstein_fast``, but starting from
    already-built n-gram count dicts. Lets callers that need to repeat this
    over many resampled subsets (e.g. bootstrap noise estimation) precompute
    per-trajectory n-gram counts once and merge them, instead of rescanning
    every transition on every resample.

    Args:
        ngram_counts1 (Counter): Precomputed n-gram counts for the first
            trajectory set.
        ngram_counts2 (Counter): Precomputed n-gram counts for the second
            trajectory set.
        global_ngrams: n-gram support to align both count dicts over.
        cost_matrix (np.ndarray): Pairwise transport cost matrix between
            n-grams in ``global_ngrams``, used by the OT solver.
        alpha (float): Additive (Laplace) smoothing applied before
            normalizing counts to probabilities. Defaults to 1e-6.

    Returns:
        float: Earth Mover's (Wasserstein) distance between the two
        smoothed n-gram distributions under ``cost_matrix``.
    """
    # 1. Map directly to the canonical global support set
    vec1 = np.array([ngram_counts1.get(k, 0) for k in global_ngrams], dtype=float)
    vec2 = np.array([ngram_counts2.get(k, 0) for k in global_ngrams], dtype=float)

    # 2. Additive Smoothing and Normalization
    support_size = len(global_ngrams)
    total1 = np.sum(vec1)
    total2 = np.sum(vec2)

    p = (vec1 + alpha) / (total1 + alpha * support_size)
    q = (vec2 + alpha) / (total2 + alpha * support_size)

    # Enforce strict float precision for the OT solver
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 3. Compute optimal transport using the precomputed matrix
    wasserstein_dist = ot.emd2(p, q, cost_matrix)

    return float(wasserstein_dist)

def compute_ngram_wasserstein_fast(trajectories1, trajectories2, global_ngrams, cost_matrix, n=3, window_size=1, alpha=1e-6) -> float:
    """Computes the n-gram Wasserstein distance between two sets of trajectories.

    Args:
        trajectories1: First set of trajectories to extract n-gram
            histograms from.
        trajectories2: Second set of trajectories to extract n-gram
            histograms from.
        global_ngrams: n-gram support to align both histograms over.
        cost_matrix (np.ndarray): Pairwise transport cost matrix between
            n-grams in ``global_ngrams``.
        n (int): Length of the action n-grams to extract. Defaults to 3.
        window_size (int): Currently unused; reserved for future stride
            support. Defaults to 1.
        alpha (float): Additive smoothing applied before normalizing
            counts to probabilities. Defaults to 1e-6.

    Returns:
        float: Earth Mover's (Wasserstein) distance between the two
        trajectory sets' n-gram distributions.
    """
    # 1. Extract empirical histograms
    ngram_counts1 = compute_ngram_histogram(trajectories1, n, window_size)
    ngram_counts2 = compute_ngram_histogram(trajectories2, n, window_size)

    return compute_ngram_wasserstein_from_counts(ngram_counts1, ngram_counts2, global_ngrams, cost_matrix, alpha)

def compute_perplexity_from_counts(counts:Dict[Any, float])->float:
    """Computes the perplexity of a distribution given raw counts.

    Args:
        counts (Dict[Any, float]): Raw counts for the distribution, keyed
            by outcome.

    Returns:
        float: Perplexity (2 ** entropy in bits) of the distribution.
    """
    vec_counts = np.array(list(counts.values()))
    h = entropy(vec_counts, base=2)
    perplexity = 2 ** h
    return perplexity
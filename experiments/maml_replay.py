import sys
from utils.trajectory_utils import load_trajectories_from_json
from trajectory import EmpiricalPolicy
from utils.aidojo_utils import aidojo_state_str_from_dict, aidojo_action_type_from_dict
from utils.trajectory_utils import policy_comparison, get_steps_for_state
from AIDojoCoordinator.game_components import ActionType, GameState, IP, Network, Service, Data
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
import collections
import argparse
import os
import re
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
from itertools import combinations_with_replacement, combinations

WANDB_PROJECT = "maml-generalization"
WANDB_ENTITY = "ondrej-lukas-czech-technical-university-in-prague"


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
                                                         state_encoder=aidojo_state_str_from_dict)
    return trajectories, metadata

def build_empirical_policy(path, max_trajectories)-> (EmpiricalPolicy, list):
    """
    Builds an EmpiricalPolicy from trajectories stored in a JSON file.
    Args:
        path (str): Path to the JSON file containing trajectories.
        max_trajectories (int): Maximum number of trajectories to load.
    Returns:
        EmpiricalPolicy: The constructed empirical policy.
        list: List of loaded trajectories.
    """
    # load the trajectories from file
    print(f"[Trajectory processing & EP build] {path}")
    trajectories, _ = load_trajectories(path, max_trajectories=max_trajectories, load_metadata=False)
    empirical_policy = EmpiricalPolicy(trajectories)
    return empirical_policy, trajectories

def collect_trajectory_data(data:dict ,max_trajectories)->dict:
    """
    Collects trajectory data and builds empirical policies in parallel.
    Args:
        data (dict): Nested dictionary with structure {checkpoint: {task_key: {pre_adaptation_path, post_adaptation_path}}}
        max_trajectories (int): Maximum number of trajectories to load per policy.
    Returns:
        dict: Nested dictionary with empirical policies added {checkpoint: {task_key: {pre_adapt_policy, post_adapt_policy}}}
    """
    # prepare paths correctly
    paths = []
    results = {}
    for cp in sorted(data.keys()):
        results[cp] = {}
        for task_key in sorted(data[cp].keys()):
            results[cp][task_key] = {}
            paths.append((cp, task_key, "pre_adapt", data[cp][task_key]["pre_adaptation_path"]))
            paths.append((cp, task_key, "post_adapt",data[cp][task_key]["post_adaptation_path"]))
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(build_empirical_policy, path, max_trajectories): (cp, task, policy_id, path)
            for (cp, task, policy_id, path) in paths
        }
        for f in as_completed(futures):
            (cp, task, policy_id, _) = futures[f]
            empirical_policy, _ = f.result()
            results[cp][task][f"{policy_id}_policy"] = empirical_policy
    return results

def get_transition_probabilities(policy: EmpiricalPolicy):
    """
    Extracts P(next_state | state) for the entire graph based on empirical counts.
    Returns: dict[state] -> list of (next_state, probability)
    """
    transitions = collections.defaultdict(lambda: collections.defaultdict(int))
    state_totals = collections.defaultdict(int)

    # 1. Aggregate counts from the edge_count map
    # structure: (state, action, next_state) -> count
    for (s, a, next_s), count in policy._edge_count.items():
        transitions[s][next_s] += count
        state_totals[s] += count

    # 2. Normalize to probabilities
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


# def _refine_cost_matrix_recursively(policy1, policy2, nodes1, nodes2, 
#                                     n1_idx, n2_idx, initial_cost_matrix, 
#                                     max_iterations=3, gamma=0.9, max_delta=1e-4):
#     """
#     Performs the fixed-point iteration to refine the cost matrix based on 
#     future transition probabilities (Lookahead).
#     """
#     n1_len = len(nodes1)
#     n2_len = len(nodes2)
    
#     # We work on a copy so we don't mutate the input immediately
#     cost_matrix = initial_cost_matrix.copy()

#     # 1. Precompute transitions: list of (child_index, prob)
#     # This optimization moves the dictionary lookup outside the main loops
#     trans1_idx = _precompute_transitions(policy1, nodes1, n1_idx)
#     trans2_idx = _precompute_transitions(policy2, nodes2, n2_idx)

#     print(f"[Cost Matrix Refining] Starting Fixed-Point Iteration (max_iteration={max_iterations})")

#     # 2. The Fixed-Point Iteration Loop
#     for k in range(max_iterations):
#         next_cost_matrix = cost_matrix.copy()
#         max_change = 0.0
#         for i in range(n1_len):
#             t_u = trans1_idx[i]       
#             for j in range(n2_len):
#                 t_v = trans2_idx[j]
                
#                 # If either node is terminal (no transitions), stable cost is just local cost
#                 if not t_u or not t_v:
#                     continue
#                 # Calculate Future Cost (Greedy Wasserstein Approx)
#                 # U -> V direction
#                 cost_u_v = 0.0
#                 for child_u_idx, prob_u in t_u:
#                     # Find closest child in V using CURRENT cost matrix
#                     min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_v_idx, _ in t_v])
#                     cost_u_v += prob_u * min_d
                    
#                 # V -> U direction (Symmetry)
#                 cost_v_u = 0.0
#                 for child_v_idx, prob_v in t_v:
#                     min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_u_idx, _ in t_u])
#                     cost_v_u += prob_v * min_d
                
#                 future_term = max(cost_u_v, cost_v_u)
                
#                 # Update Equation: Blend Current Estimate with Future Estimate
#                 new_val = initial_cost_matrix[i, j] + gamma * future_term

#                 next_cost_matrix[i, j] = new_val
#                 max_change = max(max_change, abs(new_val - cost_matrix[i, j]))
        
#         cost_matrix = next_cost_matrix
        
#         # Logging & Early Exit

#         if k % 10 == 0 or k == max_iterations - 1:
#             print(f"\tIteration {k}: max delta = {max_change:.6f}")
#         if max_change < 1e-4:
#             break
#     return cost_matrix

# def _precompute_transitions(policy: EmpiricalPolicy, node_list, node_to_idx):
#     """
#     Helper: Extracts (next_node, Probability) for every node.
#     Adapted for EmpiricalPolicy which stores edges as (s, a, s') tuples.
#     Args:
#         policy: The EmpiricalPolicy object.
#         node_list: List of nodes to consider.
#         node_to_idx: Map of {node: index} for fast lookup.
#     Returns:
#         dict: For each node in node_list, a list of (child_index, probability). 
#     """
#     # 1. Build an efficient adjacency map: State -> {Next_State: Count}
#     # We aggregate over actions because Bisimulation looks at state-to-state flow
#     adjacency = {}
#     for (s, a, next_s), count in policy._edge_count.items():
#         if s not in adjacency:
#             adjacency[s] = {}
#         if next_s not in adjacency[s]:
#             adjacency[s][next_s] = 0.0
#         adjacency[s][next_s] += count

#     cache = []
    
#     # 2. Iterate through the aligned node list in order
#     for u in node_list:
#         # Ensure 'u' is in the hashable format used by the policy internals
#         u_hash = policy._convert_to_hashable(u)
        
#         children_list = []
        
#         if u_hash in adjacency:
#             neighbors = adjacency[u_hash]
#             total_visits = sum(neighbors.values())
            
#             if total_visits > 0:
#                 for v_hash, count in neighbors.items():
#                     # FILTER: Only include children that are part of the alignment set
#                     # (The policy might have valid next_states that weren't passed in node_list)
#                     if v_hash in node_to_idx:
#                         prob = count / total_visits
#                         child_idx = node_to_idx[v_hash]
#                         children_list.append((child_idx, prob))
        
#         cache.append(children_list)
        
#     return cache

# def find_psm_mapping(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions,
#                      gamma=0.95, iterations=3, normalize_cost_matrix=False, REWARD_SCALE=100.0):

#     # 1. Setup Global Context
#     nodes1 = list(policy1.states)
#     nodes2 = list(policy2.states)
#     n1_len = len(nodes1)
#     n2_len = len(nodes2)
#     PAD_VALUE = 1e5

#     # Map nodes to indices for O(1) lookup
#     n1_idx = {n: i for i, n in enumerate(nodes1)}
#     n2_idx = {n: i for i, n in enumerate(nodes2)}

#     print(f"Aligning {n1_len} states vs {n2_len} states (Actions: {len(global_actions)})...")

#     # 2. Precompute Action Distributions
#     d1_map = {n: policy1.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes1}
#     d2_map = {n: policy2.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes2}

#     # 3. Initialize Cost Matrix (Local TVD)
#     cost_matrix = np.full((n1_len, n2_len), fill_value=PAD_VALUE)
#     for i, u in enumerate(nodes1):
#         for j, v in enumerate(nodes2):
#             is_terminal_1 = len(policy1._state_action_map.get(u, {})) == 0
#             is_terminal_2 = len(policy2._state_action_map.get(v, {})) == 0

#             if is_terminal_1 and is_terminal_2:
#                 r1 = policy1.get_average_value(u)
#                 r2 = policy2.get_average_value(v)
#                 cost_matrix[i, j] = np.tanh(abs(r1 - r2) / REWARD_SCALE)
#             elif is_terminal_1 or is_terminal_2:
#                 cost_matrix[i, j] = 1.0
#             else:
#                 cost_matrix[i, j] = compute_tvd(d1_map[u], d2_map[v], global_actions)

#     # 4. Convert EmpiricalPolicy -> Transition Matrices
#     def policy_to_matrix(policy, nodes, node_to_idx):
#         n = len(nodes)
#         T = np.zeros((n, n), dtype=np.float64)
#         for i, u in enumerate(nodes):
#             # get transition probabilities for state u ([(child_idx, prob), ...])
#             transitions = policy.get_target_transitions(u, normalize=True)  # returns list of (child_idx, prob)
#             for child, prob in transitions:
#                 c_idx = node_to_idx[child] if child in node_to_idx else child
#                 T[i, c_idx] = prob
#         return T

#     T1 = policy_to_matrix(policy1, nodes1, n1_idx)
#     T2 = policy_to_matrix(policy2, nodes2, n2_idx)

#     # 5. Vectorized Probabilistic Refinement
#     cost_matrix = bisimulation_metric_prob(
#         cost=cost_matrix,
#         T1=T1,
#         T2=T2,
#         gamma=gamma,
#         eps=1e-7,
#         max_iter=iterations
#     )

#     # Normalize to [0, 1] range for the Threshold Check (optional)
#     if normalize_cost_matrix:
#         scaling_factor = 1.0 / (1.0 - gamma)
#         cost_matrix = cost_matrix / scaling_factor

#     # 6. Solve the node matching problem
#     print("Finding Optimal Node Matching using Hungarian Algorithm...")
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     print(f"Matched {len(row_ind)} states.")

#     return cost_matrix, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map
def bisimulation_metric_prob_efficient(cost, T1, T2, gamma=0.95, eps=1e-7, max_iter=200):
    """
    Fully vectorized, memory-efficient fixed-point iteration for probabilistic bisimulation.
    
    Args:
        cost: (N1 x N2) initial cost matrix (local cost: TVD or reward diff)
        T1: (N1 x N1) transition matrix for policy 1
        T2: (N2 x N2) transition matrix for policy 2
        gamma: discount factor
        eps: convergence threshold
        max_iter: maximum number of iterations
    
    Returns:
        d: (N1 x N2) refined bisimulation metric
    """
    N1, N2 = cost.shape
    d = cost.astype(np.float64).copy()
    
    # Precompute terminal nodes
    has_child1 = np.any(T1 > 0.0, axis=1)
    has_child2 = np.any(T2 > 0.0, axis=1)
    terminal_pair = (~has_child1)[:, None] | (~has_child2)[None, :]

    # Precompute children indices
    children1 = [np.where(T1[u] > 0)[0] for u in range(N1)]
    children2 = [np.where(T2[v] > 0)[0] for v in range(N2)]

    for iteration in range(max_iter):
        # --- U -> V direction ---
        future_uv = np.zeros_like(d)
        for u in range(N1):
            if len(children1[u]) == 0:
                continue
            # Extract submatrix for all v at once
            min_matrix = np.full((len(children1[u]), N2), np.inf, dtype=np.float64)
            for v in range(N2):
                if len(children2[v]) == 0:
                    continue
                min_matrix[:, v] = np.min(d[np.ix_(children1[u], children2[v])], axis=1)
            # Weighted sum over children of u
            future_uv[u, :] = T1[u, children1[u]] @ min_matrix

        # --- V -> U direction ---
        future_vu = np.zeros_like(d)
        for v in range(N2):
            if len(children2[v]) == 0:
                continue
            min_matrix = np.full((N1, len(children2[v])), np.inf, dtype=np.float64)
            for u in range(N1):
                if len(children1[u]) == 0:
                    continue
                min_matrix[u, :] = np.min(d[np.ix_(children1[u], children2[v])], axis=0)
            future_vu[:, v] = (T2[v, children2[v]] @ min_matrix.T).T

        # --- Update ---
        d_new = cost + gamma * np.maximum(future_uv, future_vu)
        d_new[terminal_pair] = cost[terminal_pair]

        # Convergence check
        max_delta = np.max(np.abs(d_new - d))
        print(f"\tIteration {iteration}: max delta = {max_delta:.6f}")
        if max_delta < eps:
            return d_new
        d[:] = d_new

    return d


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





def bisimulation_metric_optimized(cost, T1, T2, gamma=0.95, eps=1e-7, max_iter=200):
    """
    Computes the Bisimulation Metric using optimized Vectorized Fixed-Point Iteration.
    
    Implements the "Relaxed" (Hausdorff) metric:
    d(u, v) = c(u, v) + gamma * max( E_u[min_v d], E_v[min_u d] )
    
    This is O(N^2) per iteration, significantly faster than the O(N^3) Wasserstein exact method.
    
    Args:
        cost: (N1 x N2) Initial cost matrix (e.g., reward difference or TVD).
        T1: (N1 x N1) Transition matrix for Policy 1.
        T2: (N2 x N2) Transition matrix for Policy 2.
        gamma: Discount factor.
        eps: Convergence threshold.
        max_iter: Maximum iterations.
        
    Returns:
        d: (N1 x N2) The converged distance matrix.
    """
    N1, N2 = cost.shape
    d = cost.astype(np.float64).copy()
    
    # 1. Precompute Topology
    # Identify terminal states (rows with all zeros)
    has_child1 = np.any(T1 > 0.0, axis=1)
    has_child2 = np.any(T2 > 0.0, axis=1)
    
    # Mask: True if EITHER state is terminal.
    # We fix these values to 'cost' because they have no future to compare.
    terminal_mask = (~has_child1)[:, None] | (~has_child2)[None, :]
    
    # Precompute children indices to avoid np.where inside the loop
    # Storing as list of arrays is efficient for the specific slicing we do later
    children1 = [np.where(T1[u] > 0)[0] for u in range(N1)]
    children2 = [np.where(T2[v] > 0)[0] for v in range(N2)]

    # 2. Constants
    # Use a safe large value instead of np.inf to avoid (0.0 * inf = NaN)
    # The theoretical max distance is bounded by max_reward / (1-gamma). 
    # 1e6 is safe enough for standard RL scaling.
    SAFE_MAX = 1e6 

    print(f"Starting Fixed-Point Iteration (Max {max_iter})...")

    for iteration in range(max_iter):
        d_prev = d.copy()
        
        # ---------------------------------------------------------
        # Direction 1: Forward (Policies 1 -> 2)
        # "For every future state of U, how close is the NEAREST future state of V?"
        # ---------------------------------------------------------
        
        # M1 Matrix: M1[k, v] = min_{l in children(v)} d[k, l]
        # Initialize with SAFE_MAX so that if v has no children, distance is huge.
        M1 = np.full((N1, N2), SAFE_MAX, dtype=np.float64)
        
        # We iterate N2 times (columns), performing a fast N1-vectorized min reduction
        for v in range(N2):
            kids = children2[v]
            if len(kids) > 0:
                # Efficiently find the min distance from all u-children to v's specific children
                M1[:, v] = d[:, kids].min(axis=1)
        
        # Weighted expectation over U's future
        # future_uv[u, v] = Sum_k ( T1[u, k] * Min_l d[k, l] )
        future_uv = T1 @ M1
        
        # ---------------------------------------------------------
        # Direction 2: Backward (Policies 2 -> 1)
        # "For every future state of V, how close is the NEAREST future state of U?"
        # ---------------------------------------------------------
        
        # M2 Matrix: M2[u, l] = min_{k in children(u)} d[k, l]
        M2 = np.full((N1, N2), SAFE_MAX, dtype=np.float64)
        
        for u in range(N1):
            kids = children1[u]
            if len(kids) > 0:
                M2[u, :] = d[kids, :].min(axis=0)
                
        # Weighted expectation over V's future
        # Transpose T2 to match dimensions for matrix multiplication
        # Result[u, v] = Sum_l ( T2[v, l] * Min_k d[k, l] )
        future_vu = M2 @ T2.T

        # ---------------------------------------------------------
        # Update Step (Bellman Operator)
        # ---------------------------------------------------------
        
        # Hausdorff metric: take the max of the two directional discrepancies
        discrepancy = np.maximum(future_uv, future_vu)
        
        # Clamp large values (from SAFE_MAX) to avoid numerical explosion,
        # though the terminal_mask below will overwrite the bad ones anyway.
        discrepancy[discrepancy > SAFE_MAX / 2] = SAFE_MAX
        
        d_new = cost + gamma * discrepancy
        
        # Boundary Condition:
        # If one node is terminal and the other isn't, the topology mismatch is huge,
        # BUT usually, we fall back to the immediate cost (reward diff) for stability.
        d_new[terminal_mask] = cost[terminal_mask]

        # ---------------------------------------------------------
        # Convergence Check
        # ---------------------------------------------------------
        diff = np.max(np.abs(d_new - d_prev))
        d = d_new
        
        if diff < eps:
            print(f"  Converged at iteration {iteration}. Max Delta: {diff:.2e}")
            return d
            
    print(f"  Warning: Reached max_iter ({max_iter}). Final Delta: {diff:.6f}")
    return d

def find_psm_mapping(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions,
                     gamma=0.95, iterations=3, normalize_cost_matrix=False, REWARD_SCALE=100.0):
    """
    Finds Probabilistic State Mapping (PSM) between two stochastic policies.
    Uses memory-efficient vectorized bisimulation metric.
    """
    # 1. Setup nodes
    nodes1 = list(policy1.states)
    nodes2 = list(policy2.states)
    n1_len = len(nodes1)
    n2_len = len(nodes2)

    n1_idx = {n: i for i, n in enumerate(nodes1)}
    n2_idx = {n: i for i, n in enumerate(nodes2)}

    print(f"Aligning {n1_len} states vs {n2_len} states (Actions: {len(global_actions)})...")

    # 2. Precompute Action Distributions
    d1_map = {n: policy1.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes1}
    d2_map = {n: policy2.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes2}

    # 3. Initialize local cost matrix
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
                #cost_matrix[i, j] = compute_tvd(d1_map[u], d2_map[v], global_actions)
                cost_matrix[i, j] = compute_js_distance(d1_map[u], d2_map[v], global_actions)

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
    # 5. Memory-efficient bisimulation refinement
    # cost_matrix = bisimulation_metric_prob_efficient(
    #     cost=cost_matrix,
    #     T1=T1,
    #     T2=T2,
    #     gamma=gamma,
    #     eps=1e-7,
    #     max_iter=iterations
    # )
    #Phase 1 - Use efficient approximation to get close
    # cost_matrix = bisimulation_metric_optimized(
    # #     cost=cost_matrix,
    # #     T1=T1,
    # #     T2=T2,
    # #     gamma=gamma,
    # #     eps=1e-7,
    # #     max_iter=iterations
    # # )

    # # Phase 2 - Refine with exact method
    # cost_matrix = bisimulation_metric_prob_efficient(
    #     cost=cost_matrix_approx,
    #     T1=T1,
    #     T2=T2,
    #     gamma=gamma,
    #     eps=1e-7,
    #     max_iter=5
    # )

    # Normalize (optional)
    if normalize_cost_matrix:
        scaling_factor = 1.0 / (1.0 - gamma)
        cost_matrix = cost_matrix / scaling_factor

    # 6. Hungarian Algorithm for optimal node matching
    print("Finding Optimal Node Matching using Hungarian Algorithm...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(f"Matched {len(row_ind)} states.")

    return cost_matrix, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map


def compute_otsu_threshold(data, num_bins=256):
    """
    Computes the optimal threshold using Otsu's method via NumPy.
    Correctly handles probability normalization and array alignment.
    """
    # 1. Edge Case Handling
    if len(data) == 0:
        return 0.0
    
    # If the data has no variance (all values same), return that value
    if np.min(data) == np.max(data):
        return np.min(data)

    # 2. Histogram Generation
    # We do NOT use density=True. We manually normalize to ensure sum(p) = 1.0
    counts, bin_edges = np.histogram(data, bins=num_bins)
    
    # Convert to probabilities (p_i)
    # Using float to prevent integer division issues
    p = counts.astype(float) / (counts.sum() + 1e-12)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 3. Cumulative Sums (Vectorized)
    # weight1[k] = sum(p[0]...p[k])  -> Probability of Class 0 (Background)
    weight1 = np.cumsum(p)
    
    # weight2[k] = sum(p[k+1]...p[N]) -> Probability of Class 1 (Foreground)
    weight2 = 1.0 - weight1

    # 4. Cumulative Means
    # cumulative_mean[k] = sum(p[0]*mu[0] ... p[k]*mu[k])
    cumulative_mean1 = np.cumsum(p * bin_centers)
    
    # The global mean is the last value of the cumulative mean
    global_mean = cumulative_mean1[-1]
    
    # mean2[k] corresponds to the mean of the remainder class
    # Formula: (Global_Mean - Mean_Class_0_Weighted) / Weight_Class_1
    cumulative_mean2 = global_mean - cumulative_mean1

    # 5. Class Means
    # We add a tiny epsilon to avoid division by zero if weights are 0
    mean1 = cumulative_mean1 / (weight1 + 1e-12)
    mean2 = cumulative_mean2 / (weight2 + 1e-12)

    # 6. Between-Class Variance
    # sigma^2 = w1 * w2 * (mu1 - mu2)^2
    # CRITICAL FIX: Use [:-1] for ALL arrays. 
    # We drop the last element because at the very last bin, weight2 is 0.
    between_class_variance = (
        weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    )

    # 7. Find Optimal Threshold
    # Argmax gives the index k that maximizes variance
    idx = np.argmax(between_class_variance)
    
    optimal_threshold = bin_centers[idx]
    
    return optimal_threshold

def find_knee_point(sorted_costs, sensitivity=1.0):
    """
    Finds the knee with an adjustable sensitivity parameter.
    sensitivity=1.0 -> Standard Knee
    sensitivity < 1.0 -> Stricter (ignores the worst matches before calculating)
    """
    n_total = len(sorted_costs)
    if n_total < 2: return 0.25
    
    # 1. CLIP THE TAIL based on sensitivity
    # If sensitivity is 0.9, we only look at the best 90% of matches to find the curve
    cutoff_index = int(n_total * sensitivity)
    # Ensure we have at least a few points
    cutoff_index = max(cutoff_index, 2) 
    
    view = sorted_costs[:cutoff_index]
    n = len(view)
    
    # 2. Standard Knee Logic on the "Zoomed" View
    start_point = np.array([0, view[0]])
    end_point = np.array([n-1, view[-1]])
    line_vec = end_point - start_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_start = np.column_stack((np.arange(n), view)) - start_point
    vec_proj = np.outer(np.dot(vec_from_start, line_vec_norm), line_vec_norm)
    vec_perp = vec_from_start - vec_proj
    dist = np.sqrt(np.sum(vec_perp**2, axis=1))
    
    knee_idx = np.argmax(dist)
    return view[knee_idx]

def find_policy_overlap(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions, max_refining_iterations=10, similarity_threshold=0.25):
    """
    Computes the policy overlap between two empirical policies using PSM.
    Returns detailed overlap statistics.
    """
    # 1. Get the Structural Alignment
    cost_matrix, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map = find_psm_mapping(
        policy1, policy2, global_actions,
        gamma=0.9,      # Use high gamma to align based on long-term structure
        iterations=max_refining_iterations   # Ensure convergence
    )
    
    matched_nodes_all = []
    for i, j in zip(row_ind, col_ind):
        if i >= len(nodes1) or j >= len(nodes2):
            continue
        u = nodes1[i]
        v = nodes2[j]
        cost = cost_matrix[i, j]
        matched_nodes_all.append((u, v, cost))

    node_intersection = set(policy1.states).intersection(policy2.states)
    # edge_intersection = set(policy1._edge_count.keys()).intersection(policy2._edge_count.keys())

    # compute threshold for the stability 
    all_costs = cost_matrix[row_ind, col_ind]


    # --- STRICT CALIBRATION START ---
    # Step A: Define the "Temperature" (Scale) based on the Top 20% best matches.
    # This ensures we calibrate our ruler against the "best" data we have.
    T = np.percentile(all_costs, 20) + 1e-6

    # Step B: Set a Probability Cutoff (Confidence). 
    # We demand at least 50% confidence relative to the strict T.
    # Formula: Threshold = -T * ln(Probability)
    # If T=0.1 and we want 50% confidence, Threshold is approx 0.069
    min_confidence = 0.3 
    similarity_threshold_computed = -T * np.log(min_confidence)

    # Step C: Safety Clamp
    # Ensure we don't accidentally clamp to 0.0 if data is perfect, 
    # but also don't force it to be loose (removed the 0.05 floor).
    similarity_threshold = max(similarity_threshold_computed, 1e-4)
    similarity_threshold = 1

    print(f"Stats: min_cost={np.min(all_costs):.4f}, T(10%)={T:.4f}")
    print(f"Strict Threshold: {similarity_threshold:.4f} (at {int(min_confidence*100)}% conf)")
    matched_nodes_filtered =[]
    matching_precision_list = [] 
    exact_matches = []
    for (u, v, cost) in matched_nodes_all:
        if cost <= similarity_threshold:
            matched_nodes_filtered.append((u, v, cost))
            # compute false positives and false negatives if there is any real intersection
            if len(node_intersection) > 0:
                if u == v:
                    exact_matches.append((u, v, cost))
                if u in node_intersection or v in node_intersection:
                    matching_precision_list.append((u, v, cost))
    print(f"kept {len(matched_nodes_filtered)} / {len(matched_nodes_all)} matches.")
    return matched_nodes_all, matched_nodes_filtered, matching_precision_list, exact_matches, similarity_threshold

def map_maml_trajectory_paths(base_dir_path: str, filename_pattern) -> dict:
    """
    Traverses the directory structure and maps the file paths to the 
    target nested dictionary structure.
    
    Args:
        base_dir_path: The path to the directory containing the 'cpXXX' folders.
        
    Returns:
        A nested dictionary with file paths as the final values.
    """
    data = {}
    
    print(f"Starting path mapping from: {base_dir_path}")

    # 1. Iterate through checkpoint folders (e.g., 'cp350')
    for checkpoint_id in os.listdir(base_dir_path):
        checkpoint_path = os.path.join(base_dir_path, checkpoint_id)

        # Ensure it's a directory and starts with 'cp'
        if not os.path.isdir(checkpoint_path) or not checkpoint_id.startswith('cp'):
            continue  
        # Initialize the checkpoint dictionary
        data[checkpoint_id] = {}

        # 2. Iterate through files (trajectories) inside the checkpoint folder
        for filename in os.listdir(checkpoint_path):
            file_path = os.path.join(checkpoint_path, filename)
            
            # Use the regex to match the filename and extract parameters
            match = filename_pattern.match(filename)
            
            if not match:
                continue # Skip files that don't match the expected naming convention
            
            # 3. Extract parameters using the named groups from the regex
            metadata = match.groupdict()
            
            # Determine keys for the nested dictionary
            task_id = metadata['task_id'] 
            task_key = f"task-{task_id}"

            adapt_type = metadata['adapt_type'] # 'pre' or 'post'
            
            # Determine the final key for the path (pre_adaptation_path or post_adaptation_path)
            path_key = f"{adapt_type}_adaptation_path" 

            # Initialize the task entry if it doesn't exist
            if task_key not in data[checkpoint_id]:
                data[checkpoint_id][task_key] = {}
            
            # 4. Assign the ABSOLUTE FILE PATH to the correct key
            # Using absolute path is generally safer for later file loading.
            data[checkpoint_id][task_key][path_key] = os.path.abspath(file_path)
            
            # print(f"  Mapped {path_key} for {task_key}") # Uncomment for verbose task logging
                
    return data

def run_pre_post_adaptation_comparison(data:dict, max_iterations_cm_refinement, global_actions, similarity_threshold, store_to_wandb=False, task_id=None):
    # prepare the data structure for results
    results = {
        cp:{
            "pre_adapt":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "post_adapt":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "checkpoint_stats":{
                "total_mapping_cost": [],
                "filtered_mapping_cost": [],
                "computed_similarity_threshold": [],
                "filtered_matched_nodes_count": [],
                "all_matched_nodes_count": [],
                "precise_matching_count": [],
            },

        }
        for cp in data.keys()
    }
    print("Starting pre/post adaptation comparison.")
    
    # prepare the policy pairs for comparions
    task_data = []
    for cp in sorted(data.keys()):
        for task_key in sorted(data[cp].keys()):
            pre_adapt_policy = data[cp][task_key]["pre_adapt_policy"]
            post_adapt_policy = data[cp][task_key]["post_adapt_policy"]
            task_data.append((cp, task_key, pre_adapt_policy, post_adapt_policy))

            # store stats of pre/post adaptation policies
            results[cp]["pre_adapt"]["return"].append(pre_adapt_policy.mean_return)
            results[cp]["pre_adapt"]["unique_epg_nodes"].append(pre_adapt_policy.num_states)
            results[cp]["pre_adapt"]["unique_epg_edges"].append(len(pre_adapt_policy.edges))
            results[cp]["post_adapt"]["return"].append(post_adapt_policy.mean_return)
            results[cp]["post_adapt"]["unique_epg_nodes"].append(post_adapt_policy.num_states)
            results[cp]["post_adapt"]["unique_epg_edges"].append(len(post_adapt_policy.edges))

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(find_policy_overlap, policy1, policy2, global_actions,max_iterations_cm_refinement,similarity_threshold): (cp, task, policy1, policy2)
            for (cp, task, policy1, policy2) in task_data
        }
        for f in as_completed(futures):
            (cp, task, _, _) = futures[f]
            all_matched_nodes, filtered_matched_nodes, matching_precision_list, exact_matches, computed_threshold = f.result()
            # compute stable part mapping
            total_matching_cost = sum([cost for (_, _, cost) in all_matched_nodes])
            filtered_matching_cost = sum([cost for (_, _, cost) in filtered_matched_nodes])

            results[cp]["checkpoint_stats"]["total_mapping_cost"].append(total_matching_cost)
            results[cp]["checkpoint_stats"]["filtered_mapping_cost"].append(filtered_matching_cost)
            results[cp]["checkpoint_stats"]["computed_similarity_threshold"].append(computed_threshold)
            results[cp]["checkpoint_stats"]["filtered_matched_nodes_count"].append(len(filtered_matched_nodes))
            results[cp]["checkpoint_stats"]["all_matched_nodes_count"].append(len(all_matched_nodes))
            results[cp]["checkpoint_stats"]["precise_matching_count"].append(len(matching_precision_list))

            with open(f"inner_loop_policy_overlap_{cp}_{task}.txt", "w") as fp:
                fp.write("All Matched Nodes:\n")
                for n1,n2,cost in all_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nFiltered Matched Nodes (Stable Part):\n")
                for n1,n2,cost in filtered_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nPrecise Matching Nodes:\n")
                for n1,n2,cost in matching_precision_list:
                    fp.write(f"{n1};{n2};{cost}\n") 
                fp.write("\nExact Matches:\n")
                for n1,n2,cost in exact_matches:
                    fp.write(f"{n1};{n2};{cost}\n")
    
    if store_to_wandb:
        for cp in results.keys():
            checkpoint_int = int(cp.strip("cp"))
            metrics_to_log = {}
            for phase, phase_data in results[cp].items():
                if "hist" in phase:
                    continue
                # 2. Process Pre-Adapt Stats
                for stat_key, values in phase_data.items():
                    # Using f-strings with '/' allows W&B to group them automatically in the UI
                    base_key = f"{phase}/{stat_key}" 
                    metrics_to_log.update({
                        f"{base_key}/mean": np.mean(values),
                        f"{base_key}/std":  np.std(values),

                    })   
            wandb.log(metrics_to_log, step=checkpoint_int)

    return results

def run_inter_checkpoint_comparison(data:dict, max_iterations_cm_refinement, global_actions, similarity_threshold, policy_type,store_to_wandb=False):
    # prepare the data structure for results
    results = {
        cp:{
            "checkpoint_stats":{
                "total_structural_cost": [],
                "matched_nodes": [],
                "discarded_bad_cost": [],
                "discarded_ghosts": [],
                "total_mass_overlap": [],
                "matched_edges_count": [],
                "avg_behavior_similarity": [],
                "coverage": [],
                "shared_edges":[],
                "shared_nodes":[],
                "total_nodes":[]
            },
            "stable_hist1":[],
            "stable_hist2":[]
        }
        for cp in data.keys()
    }
    results = {
        cp:{
            "policy1":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "policy2":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "checkpoint_stats":{
                "total_mapping_cost": [],
                "filtered_mapping_cost": [],
                "computed_similarity_threshold": [],
                "filtered_matched_nodes_count": [],
                "all_matched_nodes_count": [],
                "precise_matching_count": [],
                "shared_edges":[],
                "shared_nodes":[],
            },

        }
        for cp in data.keys()
    }
    print("Starting pre/post adaptation comparison.")
    # prepare the policy pairs for comparions
    # compare all pairs of tasks between the currect cp and the final cp
    task_data = []
    final_cp = sorted(data.keys())[-1]
    for cp in sorted(data.keys()):
        for task_key1 in data[cp].keys():
            for task_key2 in data[final_cp].keys():
                policy1 = data[cp][task_key1][policy_type]
                policy2 = data[final_cp][task_key2][policy_type]
                task_data.append((cp, task_key1, task_key2, policy1, policy2))

                # find direct overlap between the two policy graphs
                node_overlap = set(policy1.states) & set(policy2.states)
                edge_overlap = set(policy1.edges) & set(policy2.edges)

                # run the comparison
                results[cp]["policy1"]["return"].append(policy1.mean_return)
                results[cp]["policy1"]["unique_epg_nodes"].append(policy1.num_states)
                results[cp]["policy1"]["unique_epg_edges"].append(len(policy1.edges))
                results[cp]["policy2"]["return"].append(policy2.mean_return)
                results[cp]["policy2"]["unique_epg_nodes"].append(policy2.num_states)
                results[cp]["policy2"]["unique_epg_edges"].append(len(policy2.edges))

                results[cp]["checkpoint_stats"]["shared_nodes"].append(len(node_overlap))
                results[cp]["checkpoint_stats"]["shared_edges"].append(len(edge_overlap))
           
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(find_policy_overlap, policy1, policy2, global_actions,max_iterations_cm_refinement,similarity_threshold): (cp, task, task2, policy1, policy2)
            for (cp, task,task2, policy1, policy2) in task_data
        }
        for f in as_completed(futures):
            (cp, task, task_2, _,_) = futures[f]
            all_matched_nodes, filtered_matched_nodes, matching_precision_list, exact_matches, computed_threshold = f.result()
            total_matching_cost = sum([cost for (_, _, cost) in all_matched_nodes])
            filtered_matching_cost = sum([cost for (_, _, cost) in filtered_matched_nodes])

            results[cp]["checkpoint_stats"]["total_mapping_cost"].append(total_matching_cost)
            results[cp]["checkpoint_stats"]["filtered_mapping_cost"].append(filtered_matching_cost)
            results[cp]["checkpoint_stats"]["computed_similarity_threshold"].append(computed_threshold)
            results[cp]["checkpoint_stats"]["filtered_matched_nodes_count"].append(len(filtered_matched_nodes))
            results[cp]["checkpoint_stats"]["all_matched_nodes_count"].append(len(all_matched_nodes))
            results[cp]["checkpoint_stats"]["precise_matching_count"].append(len(matching_precision_list))

            with open(f"outer_loop_policy_overlap_{cp}-{task}_{final_cp}-{task_2}.txt", "w") as fp:
                fp.write("All Matched Nodes:\n")
                for n1,n2,cost in all_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nFiltered Matched Nodes (Stable Part):\n")
                for n1,n2,cost in filtered_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nPrecise Matching Nodes:\n")
                for n1,n2,cost in matching_precision_list:
                    fp.write(f"{n1};{n2};{cost}\n") 
                fp.write("\nExact Matches:\n")
                for n1,n2,cost in exact_matches:
                    fp.write(f"{n1};{n2};{cost}\n")
            
            # compute stable part mapping
    if store_to_wandb:
        for cp in results.keys():
            checkpoint_int = int(cp.strip("cp"))
            metrics_to_log = {}
            for phase, phase_data in results[cp].items():
                for stat_key, values in phase_data.items():
                    # Using f-strings with '/' allows W&B to group them automatically in the UI
                    base_key = f"{phase}/{stat_key}" 
                    metrics_to_log.update({
                        f"{base_key}/mean": np.mean(values),
                        f"{base_key}/std":  np.std(values),
        
                    })
            wandb.log(metrics_to_log, step=checkpoint_int)


    return results

def run_intra_checkpoint_comparison_between_task(data:dict, max_iterations_cm_refinement, global_actions, similarity_threshold, policy_type,store_to_wandb=False):
    # prepare the data structure for results
    results = {
        cp:{
            "policy1":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "policy2":{
                "return":[],
                "unique_epg_nodes":[],
                "unique_epg_edges":[],
            },
            "checkpoint_stats":{
                "total_mapping_cost": [],
                "filtered_mapping_cost": [],
                "computed_similarity_threshold": [],
                "filtered_matched_nodes_count": [],
                "all_matched_nodes_count": [],
                "precise_matching_count": [],
                "exact_matches_count": [],
                "filtered_matching_ratio": [],
                "precise_matching_ratio": [],
                "exact_matching_ratio": [],
                "shared_edges":[],
                "shared_nodes":[],
            },

        }
        for cp in data.keys()
    }
    print("Starting pre/post adaptation comparison.")
    # prepare the policy pairs for comparions
    # for every checkpoint compare all pairs of tasks within the same checkpoint
    task_data = []
    for cp in sorted(data.keys()):
        # create all task combinations
        task_combinations = combinations(data[cp].keys(), 2)
        for task_key1, task_key2 in task_combinations:               
                policy1 = data[cp][task_key1][policy_type]
                policy2 = data[cp][task_key2][policy_type]
                task_data.append((cp, task_key1, task_key2, policy1, policy2))

                # find direct overlap between the two policy graphs
                node_overlap = set(policy1.states) & set(policy2.states)
                edge_overlap = set(policy1.edges) & set(policy2.edges)

                # run the comparison
                results[cp]["policy1"]["return"].append(policy1.mean_return)
                results[cp]["policy1"]["unique_epg_nodes"].append(policy1.num_states)
                results[cp]["policy1"]["unique_epg_edges"].append(len(policy1.edges))
                results[cp]["policy2"]["return"].append(policy2.mean_return)
                results[cp]["policy2"]["unique_epg_nodes"].append(policy2.num_states)
                results[cp]["policy2"]["unique_epg_edges"].append(len(policy2.edges))

                results[cp]["checkpoint_stats"]["shared_nodes"].append(len(node_overlap))
                results[cp]["checkpoint_stats"]["shared_edges"].append(len(edge_overlap))
           
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(find_policy_overlap, policy1, policy2, global_actions,max_iterations_cm_refinement,similarity_threshold): (cp, task, task2, policy1, policy2)
            for (cp, task,task2, policy1, policy2) in task_data
        }
        for f in as_completed(futures):
            (cp, task, task_2, _,_) = futures[f]
            all_matched_nodes, filtered_matched_nodes, matching_precision_list, exact_matches, computed_threshold = f.result()
            total_matching_cost = sum([cost for (_, _, cost) in all_matched_nodes])
            filtered_matching_cost = sum([cost for (_, _, cost) in filtered_matched_nodes])

            results[cp]["checkpoint_stats"]["total_mapping_cost"].append(total_matching_cost)
            results[cp]["checkpoint_stats"]["filtered_mapping_cost"].append(filtered_matching_cost)
            results[cp]["checkpoint_stats"]["computed_similarity_threshold"].append(computed_threshold)
            results[cp]["checkpoint_stats"]["filtered_matched_nodes_count"].append(len(filtered_matched_nodes))
            results[cp]["checkpoint_stats"]["all_matched_nodes_count"].append(len(all_matched_nodes))
            results[cp]["checkpoint_stats"]["precise_matching_count"].append(len(matching_precision_list))
            results[cp]["checkpoint_stats"]["exact_matches_count"].append(len(exact_matches))
            # compute ratios
            total_matching_nodes = len(all_matched_nodes) if len(all_matched_nodes) > 0 else 1
            results[cp]["checkpoint_stats"]["filtered_matching_ratio"].append(len(filtered_matched_nodes)/total_matching_nodes)
            results[cp]["checkpoint_stats"]["precise_matching_ratio"].append(len(matching_precision_list)/total_matching_nodes)
            results[cp]["checkpoint_stats"]["exact_matching_ratio"].append(len(exact_matches)/total_matching_nodes)

            with open(f"intra_checkpoint_{policy_type}_policy_overlap_{cp}-{task}-{task_2}.txt", "w") as fp:
                fp.write("All Matched Nodes:\n")
                for n1,n2,cost in all_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nFiltered Matched Nodes (Stable Part):\n")
                for n1,n2,cost in filtered_matched_nodes:
                    fp.write(f"{n1};{n2};{cost}\n")
                fp.write("\nPrecise Matching Nodes:\n")
                for n1,n2,cost in matching_precision_list:
                    fp.write(f"{n1};{n2};{cost}\n") 
                fp.write("\nExact Matches:\n")
                for n1,n2,cost in exact_matches:
                    fp.write(f"{n1};{n2};{cost}\n")
            
            # compute stable part mapping
    if store_to_wandb:
        for cp in results.keys():
            checkpoint_int = int(cp.strip("cp"))
            metrics_to_log = {}
            for phase, phase_data in results[cp].items():
                for stat_key, values in phase_data.items():
                    # Using f-strings with '/' allows W&B to group them automatically in the UI
                    base_key = f"{phase}/{stat_key}" 
                    metrics_to_log.update({
                        f"{base_key}/mean": np.mean(values),
                        f"{base_key}/std":  np.std(values),
        
                    })
            wandb.log(metrics_to_log, step=checkpoint_int)


    return results

if __name__ == "__main__":
    # Loading the trajectories for testing
    parser = argparse.ArgumentParser(description="MAML Replay Experiment")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing trajectory files")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum number of trajectories to load per task")
    parser.add_argument("--max_iterations", type=int, default=10, help="Number of iteration for cost-matrix refinement")
    parser.add_argument("--stability_threshold", type=float, default=0.2, help="Threshold for filtering matches in the hungarian algorithm ")
    parser.add_argument(
        "--regex_pattern",
        type=str,
        #default=r"eval_epoch-\d+_task-(?P<task_id>\d+)_(?P<adapt_type>pre|post)-adapt-trajectories\.jsonl$",
        default=r"eval_epoch-\d+_task-(?P<task_id>\d+)_(?P<adapt_type>pre|post)-adapt-trajectories\.jsonl$",
        help="Regex pattern to match trajectory filenames"
    )
    parser.add_argument("--record_wandb", action="store_true", help="Whether to record the experiment with Weights and Biases")
    parser.add_argument("--task_id", action="store", help="test")

    args = parser.parse_args()

    if args.record_wandb:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=vars(args))
        print(f"W&B initialized: Project={WANDB_PROJECT}, Entity={WANDB_ENTITY}")


    global_actions = [
        ActionType.ScanNetwork, 
        ActionType.FindServices, 
        ActionType.ExploitService, 
        ActionType.FindData, 
        ActionType.ExfiltrateData
    ]

    # load the mapping of trajectory paths
    data = map_maml_trajectory_paths(args.data_dir, re.compile(args.regex_pattern))
    print(f"Found checkpoints: {list(data.keys())}")

    # collect trajectories and build policies
    policies = collect_trajectory_data(data, args.max_trajectories)

    # #COMPARISON OF pre/post adaptation policies in each checkpoint
    # comparison_results = run_pre_post_adaptation_comparison(
    #     policies,
    #     max_iterations_cm_refinement=args.max_iterations,
    #     global_actions=global_actions,
    #     similarity_threshold=0.1,
    #     store_to_wandb=args.record_wandb)
    
    # with open("inner_loop_evalatuation_result_small_env_with_mapping.json", "w") as fp:
    #     json.dump(comparison_results, fp)
    
   #COMPARISON 
    # comparison_results = run_inter_checkpoint_comparison(
    #     policies,
    #     max_iterations_cm_refinement=args.max_iterations,
    #     global_actions=global_actions,
    #     similarity_threshold=args.stability_threshold,
    #     policy_type="pre_adapt_policy",
    #     store_to_wandb=args.record_wandb)
    

    comparison_results = run_intra_checkpoint_comparison_between_task(
        policies,
        max_iterations_cm_refinement=args.max_iterations,
        global_actions=global_actions,
        similarity_threshold=args.stability_threshold,
        policy_type="post_adapt_policy",
        store_to_wandb=args.record_wandb)
    
    with open("intra_checkpoint_comparison_between_task_post_adapt_results_small_env.json", "w") as fp:
        json.dump(comparison_results, fp)
    

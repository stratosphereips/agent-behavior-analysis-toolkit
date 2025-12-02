import sys
from utils.trajectory_utils import load_trajectories_from_json
from trajectory import EmpiricalPolicy
from utils.aidojo_utils import aidojo_state_str_from_dict, aidojo_action_type_from_dict
from utils.trajectory_utils import policy_comparison
from AIDojoCoordinator.game_components import ActionType
import numpy as np
from scipy.optimize import linear_sum_assignment
import collections
import argparse
import os
import re
import wandb

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
    #print(f"[load_trajectories] Start loading from {json_file}")
    trajectories, metadata = load_trajectories_from_json(json_file, load_metadata=load_metadata, max_trajectories=max_trajectories, 
                                                         action_encoder=aidojo_action_type_from_dict,
                                                         state_encoder=aidojo_state_str_from_dict)
    #print(f"[load_trajectories] Finished loading {len(trajectories)} trajectories from {json_file}")
    return trajectories, metadata


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


def compute_future_cost(trans_u, trans_v, global_cost_matrix, nodes2_idx_map):
    """
    Approximates 1-Wasserstein distance between two transition distributions
    using a greedy weighted match (Weighted Hausdorff).
    
    Args:
        trans_u: list of (next_state_u, prob_u)
        trans_v: list of (next_state_v, prob_v)
        global_cost_matrix: The current N x M distance matrix
        nodes2_idx_map: Map of {state_v: col_index} for fast lookup
    """
    if not trans_u or not trans_v:
        # If one leads nowhere (terminal) and other continues, max penalty
        return 1.0 if (trans_u or trans_v) else 0.0

    # Forward: How far is U's future from V's future?
    cost_u_to_v = 0.0
    for next_u, prob_u in trans_u:
        # Find the "closest" state in V's future
        # (We assume the closest match in the other graph is the optimal transport target)
        min_dist = 1.0
        
        # We only check states that actually exist in the transition list
        for next_v, _ in trans_v:
            # Look up distance in the global matrix
            # If next_v isn't in nodes2 (e.g., unseen in graph 2), max penalty
            if next_v in nodes2_idx_map:
                col_idx = nodes2_idx_map[next_v]
                # We need row_idx for next_u, but trans_u might have states not in nodes1 list
                # This requires that global_cost_matrix covers all nodes. 
                # For this implementation, we assume indices align with the outer loops.
                # To make this robust, we usually pre-map everything to indices.
                # However, for the recursive step, we usually assume the node list is fixed.
                pass 
                
        # Optimization: To allow efficient lookup, we rely on the outer function
        # passing index-based transitions or handles to do the lookup.
        # See simplified implementation below.
        pass

    return 0.0 # Placeholder, see optimized version in main function

# --- MAIN METRIC FUNCTION ---

def calculate_psm_overlap(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions,
                          gamma=0.95, iterations=3, use_recursion=True):
    
    # 1. Setup Global Context
    nodes1 = list(policy1.states)
    nodes2 = list(policy2.states)
    n1_len = len(nodes1)
    n2_len = len(nodes2)
    
    # Map nodes to indices for O(1) lookup
    n1_idx = {n: i for i, n in enumerate(nodes1)}
    n2_idx = {n: i for i, n in enumerate(nodes2)}
    
    print(f"Aligning {n1_len} states vs {n2_len} states (Actions: {len(global_actions)})...")

    # 2. Pre-compute Distributions (Local Behavior)
    # Using your corrected 'get_action_distribution' method
    # Use small alpha (0.001) to keep signals sharp
    d1_map = {n: policy1.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes1}
    d2_map = {n: policy2.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes2}

    # 3. Initialize Cost Matrix (Local TVD)
    # Start with max distance (1.0 for TVD)
    cost_matrix = np.full((max(n1_len, n2_len), max(n1_len, n2_len)), fill_value=2.0)
    
    for i, u in enumerate(nodes1):
        for j, v in enumerate(nodes2):
            # Calculate L1 / TVD
            dist_sum = sum(abs(d1_map[u][a] - d2_map[v][a]) for a in global_actions)
            cost_matrix[i, j] = dist_sum * 0.5  # Normalize to [0, 1]

    # 4. Recursive Refinement (The "Lookahead" Step)
    if use_recursion and iterations > 0:
        print("Running recursive lookahead...")
        
        # Precompute transitions: list of (child_index, prob)
        # This speeds up the inner loop 100x
        trans1_idx = []
        raw_trans1 = get_transition_probabilities(policy1)
        for u in nodes1:
            # Convert node objects to matrix indices
            t_list = []
            for next_s, prob in raw_trans1.get(u, []):
                if next_s in n1_idx: # Only track if next_s is in our main list
                    t_list.append((n1_idx[next_s], prob))
            trans1_idx.append(t_list)

        trans2_idx = []
        raw_trans2 = get_transition_probabilities(policy2)
        for v in nodes2:
            t_list = []
            for next_s, prob in raw_trans2.get(v, []):
                if next_s in n2_idx:
                    t_list.append((n2_idx[next_s], prob))
            trans2_idx.append(t_list)

        # The Loop
        for k in range(iterations):
            next_cost_matrix = cost_matrix.copy()
            max_change = 0.0
            
            for i in range(n1_len):
                t_u = trans1_idx[i]
                for j in range(n2_len):
                    t_v = trans2_idx[j]
                    
                    # If either is terminal, just keep local cost
                    if not t_u or not t_v:
                        continue
                        
                    # Calculate Future Cost (Greedy Wasserstein Approx)
                    # Average distance from U's children to V's best matching children
                    
                    # U -> V direction
                    cost_u_v = 0.0
                    for child_u_idx, prob_u in t_u:
                        # Find closest child in V using CURRENT cost matrix
                        # We iterate V's children and look up dist(child_u, child_v)
                        min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_v_idx, _ in t_v])
                        cost_u_v += prob_u * min_d
                        
                    # V -> U direction (Symmetry)
                    cost_v_u = 0.0
                    for child_v_idx, prob_v in t_v:
                        min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_u_idx, _ in t_u])
                        cost_v_u += prob_v * min_d
                    
                    future_term = max(cost_u_v, cost_v_u)
                    
                    # Update Equation: Local + Gamma * Future
                    # We blend them (Paper uses sum, but for alignment we often average)
                    # Using Paper formula:
                    new_val = cost_matrix[i, j] + (gamma * future_term)
                    # Re-normalize to avoid explosion? 
                    # The paper implies d* converges. 
                    # For alignment, let's just weight it: 0.5 Local + 0.5 Future
                    new_val = 0.5 * cost_matrix[i, j] + 0.5 * future_term

                    next_cost_matrix[i, j] = new_val
                    max_change = max(max_change, abs(new_val - cost_matrix[i, j]))
            
            cost_matrix = next_cost_matrix
            if k % 10 == 0 or k == iterations - 1:
                print(f"  Iteration {k+1}: max delta = {max_change:.6f}")
            if max_change < 1e-4:
                break

    # 5. Solve Assignment (Hungarian)
    print("Solving Hungarian Assignment...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    discarded_count = 0

    # 2. Apply The Filter
    # A cost > 0.25 implies the behavior is significantly different (more than 25% divergence)
    MATCH_THRESHOLD = 0.25 

    for i, j in zip(row_ind, col_ind):
        # Only keep the match if the cost is low enough
        if cost_matrix[i, j] < MATCH_THRESHOLD:
            u = nodes1[i]
            v = nodes2[j]
            mapping[u] = v
        else:
            discarded_count += 1
    print(f"Accepted Matches: {len(mapping)}")
    print(f"Discarded (Poor) Matches: {discarded_count}")
    # 6. Final Reporting
    active_transitions = 0
    active_overlap_mass = 0.0
    real_matches_count = 0

    for u, v in mapping.items():
        # Skip Double-Ghosts
        if not policy1.has_data(u) and not policy2.has_data(v):
            continue

        real_matches_count += 1
        d1 = d1_map[u]
        d2 = d2_map[v]
        
        for action in global_actions:
            p1 = d1[action]
            p2 = d2[action]
            
            # Metric A: Mass
            active_overlap_mass += min(p1, p2)
            
            # Metric B: Edges (Using threshold)
            if p1 > 0.01 and p2 > 0.01:
                active_transitions += 1

    # Print
    print("-" * 30)
    print("PSM OVERLAP REPORT")
    print("-" * 30)
    print(f"Total Mapped Pairs:     {len(mapping)}")
    print(f"Meaningful Pairs:       {real_matches_count}")
    print(f"Active Shared Edges:    {active_transitions}")
    print(f"Total Mass Overlap:     {active_overlap_mass:.4f}")
    
    if real_matches_count > 0:
        avg = active_overlap_mass / real_matches_count
        print(f"Avg Behavior Similarity: {avg:.2%}")
    else:
        avg = 0.0
        print("Avg Behavior Similarity: 0.00%")
        
    return mapping, real_matches_count, active_transitions, active_overlap_mass, avg

def calculate_tvd(dist1:dict, dist2:dict)->float:
    """
    Calculate the Total Variation Distance (TVD) between two action distributions.
    Args:
        dist1 (dict): The first action distribution.
        dist2 (dict): The second action distribution.
    Returns:
        float: The TVD between the two policies in the given state.
    """
    all_actions = set(dist1.keys()) | set(dist2.keys())
    tvd = 0.0
    for action in all_actions:
        p1 = dist1.get(action, 0.0)
        p2 = dist2.get(action, 0.0)
        tvd += abs(p1 - p2)
    tvd *= 0.5
    return tvd

def find_epg_overlap(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy)->None:
    # Prepare cost matrix
    nodes1 = list(policy1.states)
    nodes2 = list(policy2.states)
    n1_len = len(nodes1)
    n2_len = len(nodes2)
    size = max(n1_len, n2_len)
    cost_matrix = np.full((size, size), fill_value=10)
    all_actions = [ActionType.ScanNetwork, ActionType.FindServices, ActionType.ExploitService, ActionType.FindData, ActionType.ExfiltrateData]
    
    # Precompute action distributions for all states
    dist1_map = {n:policy1.get_action_distribution(n,all_actions) for n in nodes1}
    dist2_map = {n:policy2.get_action_distribution(n,all_actions) for n in nodes2}
    
    # Fill cost matrix with TVD values
    for i in range(n1_len):
        for j in range(n2_len):
            u = nodes1[i]
            v = nodes2[j]
            if dist1_map[u] is None or dist2_map[v] is None:
                continue
            cost_matrix[i, j] = calculate_tvd(dist1_map[u], dist2_map[v])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if i < n1_len and j < n2_len:
            # # We only keep matches if the cost is reasonable.
            # # If cost is near 2.0, it means the states share NO actions.
            # if cost_matrix[i, j] < 0.2: 
            mapping[nodes1[i]] = nodes2[j]
    
    active_transitions = 0
    active_overlap_mass = 0.0
    real_matches_count = 0  # <--- Added this to track denominator

    for u, v in mapping.items():
        # 1. Skip Double-Ghosts (Terminal states in both graphs)
        if not policy1.has_data(u) and not policy2.has_data(v):
            continue

        # We found a meaningful comparison!
        real_matches_count += 1
        
        # 2. Get the probability vectors (dictionaries)
        d1 = dist1_map[u]  # Renamed from p1 to d1 to avoid confusion
        d2 = dist2_map[v]
        
        for action in all_actions:
            p1 = d1[action]  # Now p1 is the float probability
            p2 = d2[action]
            
            # Metric A: Behavioral Mass (Sum of Min)
            overlap = min(p1, p2)
            active_overlap_mass += overlap
            
            # Metric B: Active Edges (Threshold Check)
            # FILTER: Only count if action is > 1% likely in BOTH
            if p1 > 0.05 and p2 > 0.05:
                active_transitions += 1

    # ... print results ...
    print("-" * 30)
    print(f"Total Mapped Pairs (High Confidence < 0.25): {len(mapping)}")
    print(f"Discarded Pairs (Cost >= 0.25): {len(row_ind) - len(mapping)}")
    print(f"Meaningful Pairs:       {real_matches_count}") # Use this for averages
    print(f"Active Shared Edges:    {active_transitions}") 
    print(f"Total Mass Overlap:     {active_overlap_mass:.4f}")

    # Calculate Average based on REAL matches only
    if real_matches_count > 0:
        avg = active_overlap_mass / real_matches_count
        print(f"Avg Behavior Similarity: {avg:.2%}")
    else:
        print("Avg Behavior Similarity: 0.00%")

def compute_full_psm(policy1, policy2, local_cost_matrix, nodes1, nodes2, gamma=0.9, iterations=5):
    """
    Computes the recursive Policy Similarity Metric (PSM) as defined in Agarwal et al. (2020).
    
    Refines the local cost matrix by checking if 'next states' are also similar.
    Equation: d*(x,y) = TVD(pi(x), pi(y)) + gamma * W1(d*)(P(.|x), P(.|y))
    """
    print(f"Refining PSM with {iterations} lookahead iterations (Gamma={gamma})...")
    
    current_cost = local_cost_matrix.copy()
    n1 = len(nodes1)
    n2 = len(nodes2)
    
    # Cache successors and transition probs to speed up the loop
    # Structure: trans_cache[node_idx] = [(next_node_idx, probability), ...]
    trans1 = _precompute_transitions(policy1, nodes1)
    trans2 = _precompute_transitions(policy2, nodes2)

    for k in range(iterations):
        next_cost = np.zeros_like(current_cost)
        diff = 0.0
        
        # Iterate over every pair of states (u, v)
        for i in range(n1):
            transitions_u = trans1[i]
            
            for j in range(n2):
                transitions_v = trans2[j]
                
                # 1. Get the Local Base Cost (The TVD we already calculated)
                local_term = local_cost_matrix[i, j]
                
                # 2. Calculate Future Term (Wasserstein / Earth Mover's Distance)
                # We need to match children of u to children of v efficiently
                if not transitions_u or not transitions_v:
                    # If either has no future, future cost is 0 (or max penalty if you prefer)
                    future_term = 0.0
                else:
                    future_term = _solve_wasserstein_approx(
                        transitions_u, 
                        transitions_v, 
                        current_cost, 
                        nodes2 # Needed to map global index back to column index
                    )
                
                # Update Eq 3 from paper 
                next_cost[i, j] = local_term + (gamma * future_term)
        
        # Check convergence
        diff = np.max(np.abs(next_cost - current_cost))
        print(f"  Iteration {k+1}: max delta = {diff:.6f}")
        current_cost = next_cost
        if diff < 1e-4:
            break
            
    return current_cost

def _precompute_transitions(policy, node_list):
    """
    Helper: Extracts (Child_Index, Probability) for every node.
    Assumes policy.graph is a DiGraph where edges have weights.
    """
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    cache = []
    
    for u in node_list:
        children = []
        total_weight = 0.0
        
        # Collect children
        for _, v, data in policy.graph.out_edges(u, data=True):
            if v in node_to_idx: # Only track if child is in our relevant node set
                w = data.get('weight', 1.0)
                children.append( (node_to_idx[v], w) )
                total_weight += w
        
        # Normalize weights to probabilities
        if total_weight > 0:
            children = [(idx, w/total_weight) for idx, w in children]
        
        cache.append(children)
    return cache

def _solve_wasserstein_approx(trans_u, trans_v, global_cost_matrix, nodes2_map):
    """
    Approximates the Wasserstein distance between next-state distributions.
    Uses a greedy matching or small Hungarian solve.
    """
    # trans_u list of: (global_row_idx, prob)
    # trans_v list of: (global_col_idx, prob)
    
    # 1. Build a mini cost matrix for just these children
    # This is small (e.g. 5x5) so it's fast.
    len_u = len(trans_u)
    len_v = len(trans_v)
    
    mini_cost = np.zeros((len_u, len_v))
    
    for r, (idx_u_child, prob_u) in enumerate(trans_u):
        for c, (idx_v_child, prob_v) in enumerate(trans_v):
            # We look up the DISTANCE between these two children 
            # from the GLOBAL current_cost matrix
            # Note: idx_v_child is a global index, we need to know its column index in cost matrix
            # For simplicity, assuming nodes1 and nodes2 logic in _precompute maps correctly.
            # In standard Hungarian setup, row=G1, col=G2.
            
            # Since global_cost is shape (n1, n2), we use the indices directly
            dist = global_cost_matrix[idx_u_child, idx_v_child]
            mini_cost[r, c] = dist
            
    # 2. Solve Optimal Transport (Simplified)
    # Since we just want a scalar score (distance), we can use linear sum assignment
    # on the weighted costs.
    # Note: Exact Wasserstein requires LP. For speed, we use "Match Best Probabilities".
    
    # FAST APPROXIMATION (Weighted Average of Best Matches):
    # For every child of U, find the closest child of V.
    cost_u_to_v = 0.0
    for r in range(len_u):
        # Find min distance to any child of V
        min_d = np.min(mini_cost[r, :])
        # Weight by probability of occurring
        prob = trans_u[r][1]
        cost_u_to_v += prob * min_d
        
    # Do symmetric check (V to U) to ensure metric properties
    cost_v_to_u = 0.0
    for c in range(len_v):
        min_d = np.min(mini_cost[:, c])
        prob = trans_v[c][1]
        cost_v_to_u += prob * min_d
        
    return max(cost_u_to_v, cost_v_to_u) 


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

if __name__ == "__main__":
    # Loading the trajectories for testing
    parser = argparse.ArgumentParser(description="MAML Replay Experiment")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing trajectory files")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum number of trajectories to load per task")
    parser.add_argument(
        "--regex_pattern",
        type=str,
        default=r"eval_epoch-\d+_task-(?P<task_id>\d+)_(?P<adapt_type>pre|post)-adapt-trajectories\.jsonl$",
        help="Regex pattern to match trajectory filenames"
    )
    parser.add_argument("--record_wandb", action="store_true", help="Whether to record the experiment with Weights and Biases")

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

    # load and process trajectories for each checkpoint and task
    for checkpoint_id in sorted(data.keys()):
        tasks = data[checkpoint_id]
        print(f"Processing Checkpoint: {checkpoint_id}")
        task_policies = {}
        pre_adapt_stats = {
            "return":[],
            "unique_epg_nodes":[],
            "unique_epg_edges":[],
        }
        post_adapt_stats = {
            "return":[],
            "unique_epg_nodes":[],
            "unique_epg_edges":[],
        }
        checkpoint_stats = {
            "total_overlap_mass":[],
            "total_active_edges":[],
            "total_mapped_pairs":[],
            "meaningful_pairs":[],
            "avg_behavior_similarity":[],
        }
        for task_key in sorted(tasks.keys()):
            paths = tasks[task_key]
            print(f"  Loading Task: {task_key}")
            pre_adapt_trajectories, _ = load_trajectories(paths["pre_adaptation_path"], max_trajectories=args.max_trajectories, load_metadata=False)
            post_adapt_trajectories, _ = load_trajectories(paths["post_adaptation_path"], max_trajectories=args.max_trajectories, load_metadata=False)
            
            pre_adapt_policy = EmpiricalPolicy(pre_adapt_trajectories)
            post_adapt_policy = EmpiricalPolicy(post_adapt_trajectories)

            pre_adapt_stats["return"].append(pre_adapt_policy.mean_return)
            pre_adapt_stats["unique_epg_nodes"].append(pre_adapt_policy.num_states)
            pre_adapt_stats["unique_epg_edges"].append(len(pre_adapt_policy.edges))

            post_adapt_stats["return"].append(post_adapt_policy.mean_return)
            post_adapt_stats["unique_epg_nodes"].append(post_adapt_policy.num_states)
            post_adapt_stats["unique_epg_edges"].append(len(post_adapt_policy.edges))
            task_policies[task_key] = {
                "pre_adapt_policy": pre_adapt_policy,
                "post_adapt_policy": post_adapt_policy
            }
            print(f"    Loaded {len(pre_adapt_trajectories)} pre-adapt and {len(post_adapt_trajectories)} post-adapt trajectories.")
        
        # Now compute PSM overlap for each task
        for task_key, policies in task_policies.items():
            print(f"  Calculating PSM Overlap for Task: {task_key}")
            mapping, meaningful_pairs, active_edges, overlap_mass, avg_similarity = calculate_psm_overlap(
                task_policies[task_key]["pre_adapt_policy"],
                task_policies[task_key]["post_adapt_policy"],
                global_actions,
                gamma=0.9,
                iterations=20,
                use_recursion=True
            )
            checkpoint_stats["total_overlap_mass"].append(overlap_mass)
            checkpoint_stats["total_active_edges"].append(active_edges)
            checkpoint_stats["total_mapped_pairs"].append(len(mapping))
            checkpoint_stats["meaningful_pairs"].append(meaningful_pairs)
            checkpoint_stats["avg_behavior_similarity"].append(avg_similarity)
            print(f"  Completed PSM Overlap for Task: {task_key}")
        
        if args.record_wandb:
            checkpoint_int = int(checkpoint_id.lstrip('cp'))
            metrics_to_log = {}
            # 2. Process Pre-Adapt Stats
            for stat_key, values in pre_adapt_stats.items():
                # Using f-strings with '/' allows W&B to group them automatically in the UI
                base_key = f"pre_adapt/{stat_key}" 
                metrics_to_log.update({
                    f"{base_key}/mean": np.mean(values),
                    f"{base_key}/std":  np.std(values),
                    f"{base_key}/min":  np.min(values),
                    f"{base_key}/max":  np.max(values),
                })

            # 3. Process Post-Adapt Stats
            for stat_key, values in post_adapt_stats.items():
                base_key = f"post_adapt/{stat_key}"
                metrics_to_log.update({
                    f"{base_key}/mean": np.mean(values),
                    f"{base_key}/std":  np.std(values),
                    f"{base_key}/min":  np.min(values),
                    f"{base_key}/max":  np.max(values),
                })

            # 4. Process Checkpoint Stats
            for stat_key, values in checkpoint_stats.items():
                base_key = f"checkpoint/{stat_key}"
                metrics_to_log.update({
                    f"{base_key}/mean": np.mean(values),
                    f"{base_key}/std":  np.std(values),
                    f"{base_key}/min":  np.min(values),
                    f"{base_key}/max":  np.max(values),
                })
            # 4. Log everything once for this specific step
            wandb.log(metrics_to_log, step=checkpoint_int)
        
        # # Now compute PSM overlap for each task
        # for task_key, policies in task_policies.items():
        #     print(f"  Calculating PSM Overlap for Task: {task_key}")
        #     mapping = calculate_psm_overlap(
        #         policies["pre_adapt_policy"],
        #         policies["post_adapt_policy"],
        #         global_actions,
        #         gamma=0.9,
        #         iterations=5,
        #         use_recursion=True
        #     )
        #     print(f"  Completed PSM Overlap for Task: {task_key}")
        # print("=" * 50)

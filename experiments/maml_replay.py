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

def _refine_cost_matrix_recursively(policy1, policy2, nodes1, nodes2, 
                                    n1_idx, n2_idx, initial_cost_matrix, 
                                    max_iterations=3, gamma=0.9, max_delta=1e-4):
    """
    Performs the fixed-point iteration to refine the cost matrix based on 
    future transition probabilities (Lookahead).
    """
    n1_len = len(nodes1)
    n2_len = len(nodes2)
    
    # We work on a copy so we don't mutate the input immediately
    cost_matrix = initial_cost_matrix.copy()

    # 1. Precompute transitions: list of (child_index, prob)
    # This optimization moves the dictionary lookup outside the main loops
    trans1_idx = _precompute_transitions(policy1, nodes1, n1_idx)
    trans2_idx = _precompute_transitions(policy2, nodes2, n2_idx)

    print(f"[Cost Matrix Refining] Starting Fixed-Point Iteration (max_iteration={max_iterations})")

    # 2. The Fixed-Point Iteration Loop
    for k in range(max_iterations):
        next_cost_matrix = cost_matrix.copy()
        max_change = 0.0
        for i in range(n1_len):
            t_u = trans1_idx[i]       
            for j in range(n2_len):
                t_v = trans2_idx[j]
                
                # If either node is terminal (no transitions), stable cost is just local cost
                if not t_u or not t_v:
                    continue
                # Calculate Future Cost (Greedy Wasserstein Approx)
                # U -> V direction
                cost_u_v = 0.0
                for child_u_idx, prob_u in t_u:
                    # Find closest child in V using CURRENT cost matrix
                    min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_v_idx, _ in t_v])
                    cost_u_v += prob_u * min_d
                    
                # V -> U direction (Symmetry)
                cost_v_u = 0.0
                for child_v_idx, prob_v in t_v:
                    min_d = min([cost_matrix[child_u_idx, child_v_idx] for child_u_idx, _ in t_u])
                    cost_v_u += prob_v * min_d
                
                future_term = max(cost_u_v, cost_v_u)
                
                # Update Equation: Blend Current Estimate with Future Estimate
                new_val = cost_matrix[i, j] + gamma * future_term

                next_cost_matrix[i, j] = new_val
                max_change = max(max_change, abs(new_val - cost_matrix[i, j]))
        
        cost_matrix = next_cost_matrix
        
        # Logging & Early Exit

        if k % 10 == 0 or k == max_iterations - 1:
            print(f"\tIteration {k}: max delta = {max_change:.6f}")
        if max_change < 1e-4:
            break
    return cost_matrix

def _precompute_transitions(policy: EmpiricalPolicy, node_list, node_to_idx):
    """
    Helper: Extracts (next_node, Probability) for every node.
    Adapted for EmpiricalPolicy which stores edges as (s, a, s') tuples.
    Args:
        policy: The EmpiricalPolicy object.
        node_list: List of nodes to consider.
        node_to_idx: Map of {node: index} for fast lookup.
    Returns:
        dict: For each node in node_list, a list of (child_index, probability). 
    """
    # 1. Build an efficient adjacency map: State -> {Next_State: Count}
    # We aggregate over actions because Bisimulation looks at state-to-state flow
    adjacency = {}
    for (s, a, next_s), count in policy._edge_count.items():
        if s not in adjacency:
            adjacency[s] = {}
        if next_s not in adjacency[s]:
            adjacency[s][next_s] = 0.0
        adjacency[s][next_s] += count

    cache = []
    
    # 2. Iterate through the aligned node list in order
    for u in node_list:
        # Ensure 'u' is in the hashable format used by the policy internals
        u_hash = policy._convert_to_hashable(u)
        
        children_list = []
        
        if u_hash in adjacency:
            neighbors = adjacency[u_hash]
            total_visits = sum(neighbors.values())
            
            if total_visits > 0:
                for v_hash, count in neighbors.items():
                    # FILTER: Only include children that are part of the alignment set
                    # (The policy might have valid next_states that weren't passed in node_list)
                    if v_hash in node_to_idx:
                        prob = count / total_visits
                        child_idx = node_to_idx[v_hash]
                        children_list.append((child_idx, prob))
        
        cache.append(children_list)
        
    return cache


def find_psm_mapping(policy1: EmpiricalPolicy, policy2: EmpiricalPolicy, global_actions,
                          gamma=0.95, iterations=3):
    
    # 1. Setup Global Context
    nodes1 = list(policy1.states)
    nodes2 = list(policy2.states)
    n1_len = len(nodes1)
    n2_len = len(nodes2)
    
    # Map nodes to indices for O(1) lookup
    n1_idx = {n: i for i, n in enumerate(nodes1)}
    n2_idx = {n: i for i, n in enumerate(nodes2)}
    
    print(f"Aligning {n1_len} states vs {n2_len} states (Actions: {len(global_actions)})...")
    # 2. Precompute Action Distributions
    # Use small alpha (0.001) to keep signals sharp
    d1_map = {n: policy1.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes1}
    d2_map = {n: policy2.get_action_distribution(n, global_actions, alpha=0.001) for n in nodes2}

    # 3. Initialize Cost Matrix (Local TVD)
    # Start with max distance (1.0 for TVD)
    cost_matrix = np.full((max(n1_len, n2_len), max(n1_len, n2_len)), fill_value=100.0)
    
    for i, u in enumerate(nodes1):
        for j, v in enumerate(nodes2):
            # Calculate L1 / TVD between action distributions
            cost_matrix[i,j] = compute_tvd(d1_map[u], d2_map[v], global_actions)

    # 4. Recursive Refinement (The "Lookahead" Step)
    cost_matrix = _refine_cost_matrix_recursively(
        policy1=policy1,
        policy2=policy2,
        nodes1=nodes1,
        nodes2=nodes2,
        n1_idx=n1_idx,
        n2_idx=n2_idx,
        initial_cost_matrix=cost_matrix,
        max_iterations=iterations
    )
    # Normalize to [0, 1] range for the Threshold Check
    # The theoretical max is 1 / (1 - gamma)
    scaling_factor = 1.0 / (1.0 - gamma)
    cost_matrix_normalized = cost_matrix / scaling_factor
    # 5. Solve the node matching problem
    print("Finding Optimal Node Matching using Hungarian Algorithm...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix_normalized)
    print(len(row_ind), len(col_ind), len(nodes1), len(nodes2), len(n1_idx), len(n2_idx))
    return cost_matrix_normalized, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map
    
    # # 2. Apply The Filter
    # # A cost > 0.25 implies the behavior is significantly different (more than 25% divergence)
    # MATCH_THRESHOLD = 0.25 

    # for i, j in zip(row_ind, col_ind):
    #     # Only keep the match if the cost is low enough
    #     if cost_matrix[i, j] < MATCH_THRESHOLD:
    #         u = nodes1[i]
    #         v = nodes2[j]
    #         mapping[u] = v
    #     else:
    #         discarded_count += 1
    # print(f"Accepted Matches: {len(mapping)}")
    # print(f"Discarded (Poor) Matches: {discarded_count}")
    # # 6. Final Reporting
    # active_transitions = 0
    # active_overlap_mass = 0.0
    # real_matches_count = 0

    # for u, v in mapping.items():
    #     # Skip Double-Ghosts
    #     if not policy1.has_data(u) and not policy2.has_data(v):
    #         continue

    #     real_matches_count += 1
    #     d1 = d1_map[u]
    #     d2 = d2_map[v]
        
    #     for action in global_actions:
    #         p1 = d1[action]
    #         p2 = d2[action]
            
    #         # Metric A: Mass
    #         active_overlap_mass += min(p1, p2)
            
    #         # Metric B: Edges (Using threshold)
    #         if p1 > 0.01 and p2 > 0.01:
    #             active_transitions += 1

    # # Print
    # print("-" * 30)
    # print("PSM OVERLAP REPORT")
    # print("-" * 30)
    # print(f"Total Mapped Pairs:     {len(mapping)}")
    # print(f"Meaningful Pairs:       {real_matches_count}")
    # print(f"Active Shared Edges:    {active_transitions}")
    # print(f"Total Mass Overlap:     {active_overlap_mass:.4f}")
    
    # if real_matches_count > 0:
    #     avg = active_overlap_mass / real_matches_count
    #     print(f"Avg Behavior Similarity: {avg:.2%}")
    # else:
    #     avg = 0.0
    #     print("Avg Behavior Similarity: 0.00%")
        
    # return mapping, real_matches_count, active_transitions, active_overlap_mass, avg

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

    # 2. Structural Cost
    # (Low cost = high structural similarity, even if actions differ slightly)
    total_mapping_cost = cost_matrix[row_ind, col_ind].sum()
    
    # 3. Semantic Intersection Loop
    discarded_bad_match = 0
    discarded_ghost = 0
    similar_nodes_count = 0
    
    total_mass_overlap = 0.0
    meaningful_transitions_count = 0
    
    for i, j in zip(row_ind, col_ind):
        if i >= len(nodes1) or j >= len(nodes2):
            discarded_ghost += 1
            continue
        u = nodes1[i]
        v = nodes2[j]
        cost = cost_matrix[i, j]

        # Filter A: Is the structural match good enough?
        if cost < similarity_threshold:
            
            # Filter B: Do we actually have data for these nodes?
            if policy1.has_data(u) and policy2.has_data(v):
                similar_nodes_count += 1
                
                # Retrieve Distributions
                dist1 = d1_map[u]
                dist2 = d2_map[v]
                
                # Calculate Intersection
                for action in global_actions:
                    p1 = dist1.get(action, 0.0)
                    p2 = dist2.get(action, 0.0)
                    
                    # Mass: The amount of probability they share
                    total_mass_overlap += min(p1, p2)
                    
                    # Edges: Do they both actively take this action?
                    if p1 > 0.01 and p2 > 0.01:
                        meaningful_transitions_count += 1
            else:
                discarded_ghost += 1
        else:
            discarded_bad_match += 1

    # 4. Final Normalization
    # "When we find a match, how close is it?" (0.0 to 1.0)
    avg_match_similarity = (total_mass_overlap / similar_nodes_count) if similar_nodes_count > 0 else 0.0
    
    # "How much of the total graph did we manage to match?"
    total_nodes = max(len(nodes1), len(nodes2))
    coverage = similar_nodes_count / total_nodes

    results = {
        "total_structural_cost": total_mapping_cost,
        "matched_nodes": similar_nodes_count,
        "discarded_bad_cost": discarded_bad_match,
        "discarded_ghosts": discarded_ghost,
        "total_mass_overlap": total_mass_overlap,
        "shared_edges_count": meaningful_transitions_count,
        "avg_behavior_similarity": avg_match_similarity, # <--- The most important number
        "coverage": coverage
    }
    
    print("-" * 30)
    print(f"Overlap Analysis (Threshold {similarity_threshold})")
    print(f"  Nodes Matched: {similar_nodes_count} (Coverage: {coverage:.1%})")
    print(f"  Avg Similarity: {avg_match_similarity:.1%} (on matched nodes)")
    print(f"  Shared Edges:   {meaningful_transitions_count}")
    print("-" * 30)

    return results

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
    parser.add_argument("--max_iterations", type=int, default=10, help="Number of iteration for cost-matrix refinement")
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
        checkpoint_stats = {
            "total_structural_cost": [],
            "matched_nodes": [],
            "discarded_bad_cost": [],
            "discarded_ghosts": [],
            "total_mass_overlap": [],
            "shared_edges_count": [],
            "avg_behavior_similarity": [], # <--- The most important number
            "coverage": []
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
            task_checkpoint_stats = find_policy_overlap(
                task_policies[task_key]["pre_adapt_policy"],
                task_policies[task_key]["post_adapt_policy"],
                global_actions=global_actions,
                max_refining_iterations= args.max_iterations,
                similarity_threshold=0.25

            )
            for k,v in task_checkpoint_stats.items():
                checkpoint_stats[k].append(v)

        
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
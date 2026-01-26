import argparse
import os
import json
from netsecgame import ActionType
from concurrent.futures import ProcessPoolExecutor, as_completed
from trajectory import EmpiricalPolicy
from utils.plotting_utils import plot_action_per_step_distribution
from utils.trajectory_utils import build_empirical_policy_from_file, find_psm_mapping
from itertools import combinations
import numpy as np
def collect_src_file_paths(datadir, suffix=".jsonl"):
    data = {}
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if file.endswith(suffix):
                data[file] = os.path.join(root, file)
    return data

def collect_trajectory_data(data:dict, max_trajectories, action_space=None)->dict:
    """
    Collects trajectory data and builds empirical policies in parallel.
    Args:
        data (dict): Nested dictionary with structure {checkpoint: {task_key: {pre_adaptation_path, post_adaptation_path}}}
        max_trajectories (int): Maximum number of trajectories to load per policy.
        action_space (Iterable): Optional explicit action space.
    Returns:
        dict: Nested dictionary with empirical policies added {checkpoint: {task_key: {pre_adapt_policy, post_adapt_policy}}}
    """
    # prepare paths correctly
    paths = []
    results = {}
    for cp in sorted(data.keys()):
        results[cp] = {}
        paths.append((cp, data[cp]))
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(build_empirical_policy_from_file, path, max_trajectories, action_space): (cp,path)
            for (cp, path) in paths
        }
        for f in as_completed(futures):
            (cp, path) = futures[f]
            empirical_policy, _ = f.result()
            results[cp] = empirical_policy
    return results

if __name__ == "__main__":
    # Loading the trajectories for testing
    parser = argparse.ArgumentParser(description="MAML Replay Experiment")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing trajectory files")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum number of trajectories to load per task")
    parser.add_argument("--max_iterations", type=int, default=10, help="Number of iteration for cost-matrix refinement")
    parser.add_argument("--stability_threshold", type=float, default=0.1, help="Threshold for filtering matches in the hungarian algorithm ")
    
    args = parser.parse_args()
    global_actions = [
        ActionType.ScanNetwork, 
        ActionType.FindServices, 
        ActionType.ExploitService, 
        ActionType.FindData, 
        ActionType.ExfiltrateData
    ]
    data = collect_src_file_paths(args.data_dir, suffix=".jsonl")
    policies = collect_trajectory_data(data, args.max_trajectories, action_space=global_actions)
    
    print("Collected policies:")
    # Iterate through checkpoints and print policy statistics
    for cp in policies:
        # build empirical policy
        policy: EmpiricalPolicy = policies[cp]
        print(f"Checkpoint: {cp}, Num Actions: {policy.num_actions}, Num States: {policy.num_states}, win_rate: {policy.get_mean_winrate()}, mean_return: {policy.mean_return}")
        # Plot action per step distribution ()
        fig = plot_action_per_step_distribution(policy.trajectories, global_actions, title=f"Action Distribution - Checkpoint: {cp}")
        fig.savefig(f"action_distribution_cp_{cp.split('/')[-1]}.png", dpi=600)
    
    # Compare policies between checkpoints
    cp_pairs = combinations(policies.keys(), 2)
    for (cp1, cp2) in cp_pairs:
        policy1: EmpiricalPolicy = policies[cp1]
        policy2: EmpiricalPolicy = policies[cp2]
        cost_matrix, row_ind, col_ind, nodes1, nodes2, n1_idx, n2_idx, d1_map, d2_map = find_psm_mapping(
            policy1, policy2, global_actions,
            gamma=0.95,
            iterations=args.max_iterations,
            normalize_cost_matrix=True,
            REWARD_SCALE=100.0
        )
        stable_costs = []
        stable_matching = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < args.stability_threshold:
                stable_costs.append(cost_matrix[i, j])
                stable_matching.append({'ids': (int(i), int(j)), 'cost': float(cost_matrix[i, j]), "n_policy1": nodes1[i], "n_policy2": nodes2[j], "actions_dist1": {str(k): v for k, v in d1_map[nodes1[i]].items()}, "actions_dist2": {str(k): v for k, v in d2_map[nodes2[j]].items()}})
        print(f"Policy Similarity between {cp1} and {cp2}:")
        if len(stable_matching) == 0:
            print(f"  No stable matches found. Total matching cost:{np.sum(cost_matrix[row_ind, col_ind])}, AVG cost:{np.mean(cost_matrix[row_ind, col_ind])}+-{np.std(cost_matrix[row_ind, col_ind])}")
        else:
            print(f"  Number of stable matches: {len(stable_matching)}, stable costs: {np.sum(stable_costs)}, AVG cost:{np.mean(stable_costs)}+-{np.std(stable_costs)}")
        # for match in stable_matching:
        #     match_info = stable_matching[match]
        #     print(f"    Match: {match}, Cost: {match_info['cost']}, Node1: {match_info['n_policy1']}, Node2: {match_info['n_policy2']}")
        # # store the mapping results in file
        output_file = f"psm_mapping_{cp1.split('/')[-1]}_vs_{cp2.split('/')[-1]}.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_matching = [
                {k: int(v) if isinstance(v, np.integer) else v for k, v in match.items()}
                for match in stable_matching
            ]
            json.dump(serializable_matching, f, indent=4)


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
        print(f"Checkpoint: {cp}, Num Actions: {policy.num_actions}, Num States: {policy.num_states}, win_rate: {np.mean(policy.wins):.2f}+-{np.std(policy.wins):.2f}, mean_return: {np.mean(policy.returns):.2f}+-{np.std(policy.returns):.2f}")
        # Plot action per step distribution ()

        instance_name_list = cp.split(".jsonl")[0]
        model_name = instance_name_list.split("_")[0]
        instance_name = " ".join(instance_name_list.split("_")[1:])
        
        fig = plot_action_per_step_distribution(policy.trajectories, global_actions, title=f"Action Distribution - {model_name}({instance_name})")
        fig.savefig(f"action_distribution_cp_{model_name}_{instance_name}.png", dpi=600)

import argparse
import os
import json
import warnings
from netsecgame import ActionType
from concurrent.futures import ProcessPoolExecutor, as_completed
from trajectory import EmpiricalPolicy
from utils.plotting_utils import plot_action_per_step_distribution, plot_generalization_bidirectional, plot_action_divergence_per_step, plot_action_divergence_overlay, plot_jsd_vs_unseen_return
from utils.trajectory_utils import build_empirical_policy_from_file, find_psm_mapping, drop_empty_trajectories
from utils.metrics import compute_stepwise_action_jsd
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

def load_generalization_config(config_path: str, max_trajectories, action_space=None) -> dict:
    """
    Loads an explicit model -> {seen, unseen} trajectory-file mapping for the
    behavioral generalization figure, bypassing filename-based inference.

    Config format (JSON):
        {
          "Model Name": {"seen": "path/to/seen.jsonl", "unseen": "path/to/unseen.jsonl"},
          ...
        }

    Returns:
        dict: {model_name: {"seen": [Trajectory, ...], "unseen": [Trajectory, ...]}}
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    jobs = [
        (model_name, kind, path)
        for model_name, kinds in config.items()
        for kind, path in kinds.items()
    ]
    model_groups = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(build_empirical_policy_from_file, path, max_trajectories, action_space): (model_name, kind)
            for (model_name, kind, path) in jobs
        }
        for f in as_completed(futures):
            model_name, kind = futures[f]
            _, trajectories = f.result()
            model_groups.setdefault(model_name, {})[kind] = trajectories
    return model_groups

if __name__ == "__main__":
    # Loading the trajectories for testing
    parser = argparse.ArgumentParser(description="MAML Replay Experiment")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing trajectory files (per-checkpoint stats/plots + filename-based generalization grouping)")
    parser.add_argument("--generalization_config", type=str, default=None, help="Path to a JSON config mapping model name -> {seen, unseen} trajectory file paths, for the behavioral generalization figure. Takes precedence over --data_dir filename inference.")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Maximum number of trajectories to load per task")
    parser.add_argument("--max_iterations", type=int, default=10, help="Number of iteration for cost-matrix refinement")
    parser.add_argument("--stability_threshold", type=float, default=0.1, help="Threshold for filtering matches in the hungarian algorithm ")
    parser.add_argument("--optimal_length", type=int, default=5, help="Minimum number of steps to solve the task; controls x-axis emphasis in the generalization figure")
    parser.add_argument("--generalization_output", type=str, default="behavioral_generalization.png", help="Output path for the seen-vs-unseen behavioral generalization figure")
    parser.add_argument("--divergence_output", type=str, default="action_divergence.png", help="Output path for the seen-vs-unseen per-step action-distribution JSD figure")
    parser.add_argument("--divergence_overlay_output", type=str, default="action_divergence_overlay.png", help="Output path for the single-axes overlay of all models' per-step JSD")
    parser.add_argument("--jsd_vs_return_output", type=str, default="jsd_vs_unseen_return.png", help="Output path for the mean-JSD vs. mean-unseen-return scatter figure")
    parser.add_argument("--keep_empty_trajectories", action="store_true", help="Keep trajectories with zero recorded actions (dropped by default -- they silently skew win/loss padding in the JSD computation, see utils.trajectory_utils.drop_empty_trajectories)")


    args = parser.parse_args()
    if not args.data_dir and not args.generalization_config:
        parser.error("At least one of --data_dir or --generalization_config is required")
    global_actions = [
        ActionType.ScanNetwork, 
        ActionType.FindServices, 
        ActionType.ExploitService, 
        ActionType.FindData, 
        ActionType.ExfiltrateData
    ]
    model_groups = {}
    if args.data_dir:
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

            # Group by model, classifying each checkpoint as "seen" or "unseen" topology
            # from its instance name, for the combined behavioral generalization figure.
            # Superseded below if --generalization_config is given.
            instance_name_lower = instance_name.lower()
            if "unseen" in instance_name_lower:
                kind = "unseen"
            elif "seen" in instance_name_lower:
                kind = "seen"
            else:
                continue
            model_groups.setdefault(model_name, {})[kind] = policy.trajectories

    if args.generalization_config:
        model_groups = load_generalization_config(args.generalization_config, args.max_trajectories, action_space=global_actions)

    # Only models with both a seen and an unseen topology can be plotted side by side.
    generalization_models = {
        name: kinds for name, kinds in model_groups.items()
        if "seen" in kinds and "unseen" in kinds
    }

    if not args.keep_empty_trajectories:
        for name, kinds in generalization_models.items():
            for kind, trajectories in kinds.items():
                filtered = drop_empty_trajectories(trajectories)
                n_dropped = len(trajectories) - len(filtered)
                if n_dropped:
                    warnings.warn(
                        f"{name} ({kind}): dropped {n_dropped}/{len(trajectories)} trajectories with zero "
                        f"actions (pass --keep_empty_trajectories to disable this)"
                    )
                kinds[kind] = filtered

    if generalization_models:
        gen_fig = plot_generalization_bidirectional(generalization_models, global_actions, optimal_length=args.optimal_length)
        gen_fig.savefig(args.generalization_output, dpi=350)
        print(f"Saved behavioral generalization figure to {args.generalization_output}")

        # Numerical counterpart to the figure above: per-step JSD between the seen
        # and unseen action distributions, with early-terminating episodes padded
        # with a win/loss token so the divergence stays defined at every step.
        divergence_results = {}
        for name, kinds in generalization_models.items():
            result = compute_stepwise_action_jsd(kinds["seen"], kinds["unseen"], global_actions)
            divergence_results[name] = result
            print(f"{name}: mean action-distribution JSD (seen vs. unseen) = {result['mean_jsd']:.4f}")

        div_fig = plot_action_divergence_per_step(divergence_results)
        div_fig.savefig(args.divergence_output, dpi=350)
        print(f"Saved action divergence figure to {args.divergence_output}")

        overlay_fig = plot_action_divergence_overlay(divergence_results)
        overlay_fig.savefig(args.divergence_overlay_output, dpi=350)
        print(f"Saved action divergence overlay figure to {args.divergence_overlay_output}")

        # Does higher seen/unseen behavioral divergence track with worse unseen
        # performance? Plot each model's mean JSD against its mean unseen return.
        unseen_returns = {
            name: float(np.mean([t.total_reward() for t in kinds["unseen"]]))
            for name, kinds in generalization_models.items()
        }
        jsd_vs_return_fig = plot_jsd_vs_unseen_return(divergence_results, unseen_returns)
        jsd_vs_return_fig.savefig(args.jsd_vs_return_output, dpi=350)
        print(f"Saved mean-JSD vs. unseen-return figure to {args.jsd_vs_return_output}")
    else:
        print("No model has both 'seen' and 'unseen' trajectory files; skipping behavioral generalization figure.")

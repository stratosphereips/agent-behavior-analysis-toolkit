import argparse
import json
import csv
import os
from utils.data_utils import load_policies_from_directory
from utils.metrics import (
    topological_shift,
    strategic_shift,
    traversal_depth,
    compute_entropy_metrics,
    calculate_temporal_action_entropy,
    compute_ngram_jsd,
    compute_ngram_wasserstein_fast,
    compute_decomposed_jsd,
    compute_perplexity_from_counts
)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils.plotting_utils import plot_behavioral_ontogeny
from utils.trajectory_utils import js_divergence_per_state, compute_trajectory_surprises
import itertools
from trajectory import EmpiricalPolicy
from concurrent.futures import ProcessPoolExecutor, as_completed

def save_metrics_to_csv(data, filename, name_val, mode_val):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Iterating through the dictionary to write one line per metric
        for metric_name, values in data.items():
            # Constructing the row: name, mode, metric_name, followed by all values in the list
            row = [name_val, mode_val, metric_name] + values
            writer.writerow(row)

def checkpoint_stats_worker(policy):
    """
    Worker function for computing stats for a single checkpoint.
    Args:
        policy: Policy to compute stats for
    Returns:
        dict: Dictionary of stats for the given checkpoint
    """
    results = {}
    results["mean_return"] = np.mean(policy.returns)
    results["std_return"] = np.std(policy.returns)
    results["state_visitation_perplexity"] = compute_perplexity_from_counts(policy._state_visitation_count)
    results["total_nodes"] = len(policy._state_visitation_count)
    return results

def compute_checkpoint_stats(checkpoint_policies):
    """
    Compute stats for all checkpoints.
    Args:
        checkpoint_policies: Dictionary of policies
    Returns:
        dict: Dictionary of stats for all checkpoints
    """
    checkpoint_stats = {}
    for cp_key in checkpoint_policies:
        print(f"[Stats computation] Checkpoint: {cp_key}")
        checkpoint_stats[cp_key] = checkpoint_stats_worker(checkpoint_policies[cp_key])
    return checkpoint_stats

def checkpoint_comparison_worker(current_policy, previous_policy, ngram_cost_matrix, global_ngrams, global_actions):
    """
    Worker function for comparing checkpoints.
    Args:
        current_policy: Current policy
        previous_policy: Previous policy
        ngram_cost_matrix: Cost matrix for n-grams
        global_ngrams: Global ordering of n-grams
        global_actions: Global ordering of actions
    Returns:
        dict: Dictionary of errors for the given checkpoint pair
    """
    results = {}
    topo_shift_values =  compute_decomposed_jsd(current_policy._state_visitation_count, previous_policy._state_visitation_count)
    results["topological_shift"] = topo_shift_values["jsd_total"]
    results["topological_shift_overlap"] = topo_shift_values["jsd_overlap"]
    results["topological_shift_non_overlap"] = topo_shift_values["jsd_non_overlap"]
    results["strategic_shift"] = strategic_shift(current_policy, previous_policy, global_actions=global_actions, noise_value=0.0)
    results["3-gram_wasserstein"] = compute_ngram_wasserstein_fast(current_policy.trajectories, previous_policy.trajectories, global_ngrams, ngram_cost_matrix, n=3)
    results["node_overlap"] = len(set(current_policy.states).intersection(set(previous_policy.states)))
    results["nodes_added"] = len(set(current_policy.states).difference(set(previous_policy.states)))
    results["nodes_removed"] = len(set(previous_policy.states).difference(set(current_policy.states)))
    return results

def compare_checkpoints(checkpoint_pairs, checkpoint_policies, ngram_cost_matrix, global_ngrams, global_actions):
    """
    Compare checkpoints.
    Args:
        checkpoint_pairs: List of checkpoint pairs
        ngram_cost_matrix: Cost matrix for n-grams
        global_ngrams: Global ordering of n-grams
        global_actions: Global ordering of actions
    Returns:
        dict: Dictionary of errors for each checkpoint
    """
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(checkpoint_comparison_worker, checkpoint_policies[current], checkpoint_policies[previous], ngram_cost_matrix, global_ngrams, global_actions): (current, previous)
            for previous, current in checkpoint_pairs
        }
        
        for f in as_completed(futures):
            current, previous = futures[f]
            try:
                results[(current, previous)] = f.result()
            except Exception as e:
                print(f"Error comparing checkpoint {current} and {previous}: {e}")
    return results

def compute_errors_per_checkpoint(checkpoint_policies:dict, ngram_cost_matrix, global_ngrams, global_actions):
    """
    Compute errors for each checkpoint.
    Args:
        checkpoint_policies: Dictionary of policies for each checkpoint
        ngram_cost_matrix: Cost matrix for n-grams
        global_ngrams: Global ordering of n-grams
        global_actions: Global ordering of actions
    Returns:
        dict: Dictionary of errors for each checkpoint
    """
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(estimate_noise_value, policy.trajectories, ngram_cost_matrix, global_ngrams, global_actions): cp_key
            for (cp_key, policy) in checkpoint_policies.items()
        }
        
        for f in as_completed(futures):
            cp_key = futures[f]
            try:
                results[cp_key] = f.result()
            except Exception as e:
                print(f"Error computing errors for {cp_key}: {e}")
    return results

def estimate_noise_value(trajectories, cost_matrix, global_ngrams, global_actions, num_samples:int=100, percentile:float=0.95) -> float:
    """
    Estimate the noise value for the given trajectories.
    Args:
        trajectories: List of trajectories
    Returns:
        dict: Dictionary containaining the estimated mean noise value, its standard deviation and the threshold value for each metric based 
        on the percentile of the distribution of noise values.
    """
    N = len(trajectories)
    half_N = N // 2
    errors = {
        "topological_shift": [],
        "topological_shift_overlap": [],
        "topological_shift_non_overlap": [],
        "strategic_shift": [],
        "robust_traversal_depth": [],
        "3-gram_wasserstein": [],
        "state_visitation_perplexity": [],
    }
    for _ in range(num_samples):
        indices = np.random.permutation(N)
        set_A = [trajectories[i] for i in indices[:half_N]]
        set_B = [trajectories[i] for i in indices[half_N:2*half_N]]
        
        ep_A = EmpiricalPolicy(set_A, metadata=None)
        ep_B = EmpiricalPolicy(set_B, metadata=None)
        jsd_results = compute_decomposed_jsd(ep_A._state_visitation_count, ep_B._state_visitation_count)
        errors["topological_shift"].append(jsd_results["jsd_total"])
        errors["topological_shift_overlap"].append(jsd_results["jsd_overlap"])
        errors["topological_shift_non_overlap"].append(jsd_results["jsd_non_overlap"])
        errors["strategic_shift"].append(strategic_shift(ep_A, ep_B, global_actions=global_actions, noise_value=0.0))
        errors["robust_traversal_depth"].append(min(traversal_depth(ep_A.trajectories), traversal_depth(ep_B.trajectories)))
        #errors["3-gram_jsd"].append(compute_ngram_jsd(ep_A.trajectories, ep_B.trajectories, n=3,action_space_size=len(global_actions)))
        errors["3-gram_wasserstein"].append(compute_ngram_wasserstein_fast(ep_A.trajectories, ep_B.trajectories, global_ngrams, cost_matrix, n=3))
        errors["state_visitation_perplexity"].append(compute_perplexity_from_counts(ep_A._state_visitation_count))
        errors["state_visitation_perplexity"].append(compute_perplexity_from_counts(ep_B._state_visitation_count))
    return {
    k: {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "threshold": float(np.percentile(v, percentile))
    } 
    for k, v in errors.items()
}
def get_levenshtein_distance(seq1: tuple, seq2: tuple) -> float:
    """
    Compute the Levenshtein distance between two sequences.
    Args:
        seq1: First sequence
        seq2: Second sequence
    Returns:
        levenshtein_distance: Levenshtein distance between the two sequences
    """
    size_x, size_y = len(seq1) + 1, len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
            else:
                matrix[x, y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1] + 1, matrix[x, y-1] + 1)
    return matrix[size_x - 1, size_y - 1]

def build_global_environment_cache(action_space: list, n: int = 3):
    """
    Build the global environment cache for n-gram analysis.
    Args:
        action_space: List of actions in the environment
        n: Length of n-grams
    Returns:
        global_ngrams: Canonical ordering of all possible n-grams
        cost_matrix: Static cost matrix for n-grams
    """
    # 1. Generate canonical ordering of all possible n-grams
    global_ngrams = list(itertools.product(action_space, repeat=n))
    num_motifs = len(global_ngrams)
    
    # 2. Build the static ground cost matrix
    cost_matrix = np.zeros((num_motifs, num_motifs))
    for i in range(num_motifs):
        for j in range(i, num_motifs):
            dist = get_levenshtein_distance(global_ngrams[i], global_ngrams[j])
            cost_matrix[i, j] = dist
            cost_matrix[j, i] = dist
            
    return global_ngrams, cost_matrix

def main():
    # GLOBAL_ACTIONS is now determined dynamically
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the data")
    parser.add_argument("--max_trajectories", type=int, default=1000, help="Maximum number of trajectories to load")
    parser.add_argument("--num_actions", type=int, default=None, help="Number of actions in the environment")
    parser.add_argument("--every_nth", type=int, nargs='+', default=[1], help="List of intervals for plotting points")
    parser.add_argument("--output_prefix", type=str, default="figures/behavioral_ontogeny", help="Prefix for output image")
    
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable Weights & Biases logging")
    parser.add_argument("--use_wanndb", action="store_true", dest="use_wandb", help=argparse.SUPPRESS) # alias
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="Tags for Weights & Biases run")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Run name for Weights & Biases")

    
    args = parser.parse_args()
    
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    else:
        run_name = f"cp_comp_{args.data_dir.replace('/', '_')}"
        
    if args.use_wandb:
        import wandb
        tags = ["sequential_cp_comparison"]
        if args.wandb_tags:
            tags.extend(args.wandb_tags)
        wandb.init(
            project="agent_trajectory_analysis",
            name=run_name,
            config=vars(args),
            tags=tags
        )

    # load empirical policies from files
    policies = load_policies_from_directory(args.data_dir, args.max_trajectories, test_split=None)
    checkpoints = list(policies.keys())
    # Sort checkpoints by the integer value in the filename (assuming cp_XXXX format)
    # The dictionary keys might not be sorted correctly if they are strings like "cp_100", "cp_1000"
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else x)
    
    checkpoint_pairs = list(zip(checkpoints[:-1], checkpoints[1:]))
    
    checkpoint_labels = [int(cp.split("_")[-1]) for cp in checkpoints]

    checkpoint_policies = {}
    for cp_key in checkpoints:
        checkpoint_policies[cp_key] = policies[cp_key][0]

    
    # Determine GLOBAL_ACTIONS
    if args.num_actions is not None:
        GLOBAL_ACTIONS = list(range(args.num_actions))
        print(f"Using provided num_actions: {args.num_actions} -> {GLOBAL_ACTIONS}")
    elif len(checkpoints) > 0:
        print("num_actions not provided. Inferring from metadata/trajectories...")
        first_policy = policies[checkpoints[0]][0]
        # Try to get from metadata
        action_space_size = first_policy.metadata.get("action_space_size")
        
        if action_space_size is not None:
            GLOBAL_ACTIONS = list(range(action_space_size))
            print(f"Detected action space size from metadata: {action_space_size}")
        else:
            all_actions = set()
            for cp in checkpoints:
                all_actions.update(policies[cp][0].actions)
            GLOBAL_ACTIONS = sorted(list(all_actions))
            print(f"Inferred GLOBAL_ACTIONS from trajectories: {GLOBAL_ACTIONS}")
    else:
        GLOBAL_ACTIONS = [0, 1] # Fallback default

    # precompute ngrams and distance matrix
    global_ngrams_3, cost_matrix_3 = build_global_environment_cache(GLOBAL_ACTIONS, n=3)

    checkpoint_stats = compute_checkpoint_stats(checkpoint_policies)
    
    # compute errors for each checkpoint
    errors = compute_errors_per_checkpoint(checkpoint_policies, cost_matrix_3, global_ngrams_3, GLOBAL_ACTIONS)

    
    # compare checkpoints
    comparisons = compare_checkpoints(checkpoint_pairs, checkpoint_policies, cost_matrix_3, global_ngrams_3, GLOBAL_ACTIONS)

    # merge results
    metrics = { 
        "topological_shift_raw": [],
        "mean_return": [],
        "std_return": [],
        "state_visitation_perplexity": [],
        "topological_shift_overlap_raw": [],
        "topological_shift_non_overlap_raw": [],
        "strategic_shift_raw": [],
        "3-gram_wasserstein_raw": [],
        "topological_shift_noise_threshold": [],
        "topological_shift_overlap_noise_threshold": [],
        "topological_shift_non_overlap_noise_threshold": [],
        "strategic_shift_noise_threshold": [],
        "3-gram_wasserstein_noise_threshold": [],
        "total_nodes": [],
        "node_overlap": [],
        "checkpoint_ids": [],
        "nodes_added": [],
        "nodes_removed": [],
    }
    print(f"checkpoint_pairs: {checkpoint_pairs}")
    for i, timestep in enumerate(checkpoints):
        # add checkpoint id
        metrics["checkpoint_ids"].append(i)

        # add checkpoint stats
        metrics["mean_return"].append(checkpoint_stats[timestep]["mean_return"])
        metrics["std_return"].append(checkpoint_stats[timestep]["std_return"])
        metrics["state_visitation_perplexity"].append(checkpoint_stats[timestep]["state_visitation_perplexity"])
        metrics["total_nodes"].append(checkpoint_stats[timestep]["total_nodes"])

        # add correct deltas (comparisons with previous checkpoint)
        if i > 0:
            print(f"i: {i}")
            print(f"checkpoint_pairs[i-1]: {checkpoint_pairs[i-1]}")
            current_ts = checkpoint_pairs[i-1][1]
            prev_ts = checkpoint_pairs[i-1][0]
            # add deltas
            metrics["topological_shift_raw"].append(comparisons[current_ts, prev_ts]["topological_shift"])
            metrics["topological_shift_overlap_raw"].append(comparisons[current_ts, prev_ts]["topological_shift_overlap"])
            metrics["topological_shift_non_overlap_raw"].append(comparisons[current_ts, prev_ts]["topological_shift_non_overlap"])
            metrics["strategic_shift_raw"].append(comparisons[current_ts, prev_ts]["strategic_shift"])
            metrics["3-gram_wasserstein_raw"].append(comparisons[current_ts, prev_ts]["3-gram_wasserstein"])
            metrics["node_overlap"].append(comparisons[current_ts, prev_ts]["node_overlap"])
            metrics["nodes_added"].append(comparisons[current_ts, prev_ts]["nodes_added"])
            metrics["nodes_removed"].append(comparisons[current_ts, prev_ts]["nodes_removed"])

            # add noise threshold deltas
            metrics["topological_shift_noise_threshold"].append(errors[current_ts]["topological_shift"]["threshold"])
            metrics["topological_shift_overlap_noise_threshold"].append(errors[current_ts]["topological_shift_overlap"]["threshold"])
            metrics["topological_shift_non_overlap_noise_threshold"].append(errors[current_ts]["topological_shift_non_overlap"]["threshold"])
            metrics["strategic_shift_noise_threshold"].append(errors[current_ts]["strategic_shift"]["threshold"])
            metrics["3-gram_wasserstein_noise_threshold"].append(errors[current_ts]["3-gram_wasserstein"]["threshold"])

        if args.use_wandb:
            try:
                step_metrics = {"checkpoint": checkpoint_labels[i]}
                for k, v in metrics.items():
                    if len(v) > 0:
                        step_metrics[k] = v[-1]
                wandb.log(step_metrics, step=checkpoint_labels[i])
            except Exception as e:
                print(f"Wandb logging error: {e}")
    
    # print("\n--- Metric Correlations ---")
    # corr_metrics = [
    #     "topological_shift", "strategic_shift", "robust_traversal_depth", 
    #     "reward", "effective_state_coverage", "empirical_policy_certainty", 
    #     "unweighted_policy_certainty", "temporal_action_entropy"
    # ]
    # valid_metrics = [m for m in corr_metrics if len(metrics.get(m, [])) > 0]
    
    # if len(valid_metrics) > 1:
    #     min_len = min(len(metrics[m]) for m in valid_metrics)
    #     if min_len > 1:
    #         aligned_data = np.array([metrics[m][-min_len:] for m in valid_metrics])
            
    #         # Suppress warnings for constant data resulting in NaNs
    #         with np.errstate(invalid='ignore'):
    #             pearson_corr = np.corrcoef(aligned_data)
    #             spearman_corr, _ = stats.spearmanr(aligned_data, axis=1)
            
    #         print("Pearson Correlation:")
    #         header = f"{'':>28} " + " ".join([f"{m[:8]:>8}" for m in valid_metrics])
    #         print(header)
    #         for i, m1 in enumerate(valid_metrics):
    #             row_str = " ".join([f"{pearson_corr[i,j]:8.3f}" for j in range(len(valid_metrics))])
    #             print(f"{m1:>28} {row_str}")
                
    #         print("\nSpearman Correlation:")
    #         print(header)
    #         for i, m1 in enumerate(valid_metrics):
    #             row_str = " ".join([f"{spearman_corr[i,j]:8.3f}" for j in range(len(valid_metrics))])
    #             print(f"{m1:>28} {row_str}")

    #         if args.use_wandb:
    #             def log_corr_matrix(matrix, title, tag):
    #                 try:
    #                     fig, ax = plt.subplots(figsize=(10, 8))
    #                     # Use np.nan_to_num to avoid matshow issues with pure NaN
    #                     clean_matrix = np.nan_to_num(matrix, nan=0.0)
    #                     cax = ax.matshow(clean_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    #                     fig.colorbar(cax)
                        
    #                     for i in range(len(valid_metrics)):
    #                         for j in range(len(valid_metrics)):
    #                             val = matrix[i,j]
    #                             text_val = f"{val:.2f}" if not np.isnan(val) else "NaN"
    #                             text_color = 'white' if abs(clean_matrix[i,j]) >= 0.5 else 'black'
    #                             ax.text(j, i, text_val, ha='center', va='center', color=text_color)
                                
    #                     ax.set_xticks(range(len(valid_metrics)))
    #                     ax.set_yticks(range(len(valid_metrics)))
    #                     ax.set_xticklabels(valid_metrics, rotation=45, ha='left')
    #                     ax.set_yticklabels(valid_metrics)
    #                     plt.title(title, pad=20)
    #                     plt.tight_layout()
    #                     wandb.log({tag: wandb.Image(fig)})
    #                     plt.close(fig)
    #                 except Exception as e:
    #                     print(f"Wandb {tag} logging error: {e}")
                        
    #             log_corr_matrix(pearson_corr, "Pearson Correlation", "correlations/pearson")
    #             log_corr_matrix(spearman_corr, "Spearman Correlation", "correlations/spearman")

    # Visualization of experiments
    try:
        fig, axes = plt.subplots(5, 1, figsize=(12, 25), sharex=True)
        fig.suptitle(f"Run: {run_name}", fontsize=20)
        
        # Plot 1: Reward
        axes[0].plot(checkpoint_labels, metrics["mean_return"], label="Mean Return", marker='x')
        axes[0].fill_between(checkpoint_labels, 
                             np.array(metrics["mean_return"]) - np.array(metrics["std_return"]),
                             np.array(metrics["mean_return"]) + np.array(metrics["std_return"]),
                             alpha=0.3)
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Model Reward")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)

        # Plot 2: Node Counts
        x_deltas = checkpoint_labels[1:]
        axes[1].plot(checkpoint_labels, metrics["total_nodes"], label="Total Visited Nodes", marker='x', color='blue')
        axes[1].plot(x_deltas, metrics["node_overlap"], label="Overlapping Nodes", marker='x', color='orange')
        axes[1].plot(x_deltas, metrics["nodes_added"], label="Added Nodes", marker='x', color='green')
        axes[1].plot(x_deltas, metrics["nodes_removed"], label="Removed Nodes", marker='x', color='red')
        axes[1].set_ylabel("Count")
        axes[1].set_title("Node Discovery and Overlap")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)

        # Plot 3: Perplexity
        axes[2].plot(checkpoint_labels, metrics["state_visitation_perplexity"], label="State Visitation Perplexity", marker='x', color='purple')
        axes[2].plot(checkpoint_labels, metrics["total_nodes"], label="Max Possible Perplexity (Nodes Visited)", linestyle='--', color='gray', alpha=0.7)
        axes[2].set_ylabel("Perplexity")
        axes[2].set_title("State Visitation Perplexity")
        max_nodes = max(metrics["total_nodes"]) if metrics["total_nodes"] else 1
        axes[2].set_ylim(0, max_nodes * 1.05)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True)

        # Plot 4: Shifts
        axes[3].plot(x_deltas, metrics["topological_shift_raw"], label="Full Topological Shift", marker='x')
        axes[3].plot(x_deltas, metrics["topological_shift_overlap_raw"], label="Topological Shift on Overlap", marker='x')
        axes[3].plot(x_deltas, metrics["strategic_shift_raw"], label="Strategic Shift", marker='x')
        axes[3].set_ylabel("Value (JSD)")
        axes[3].set_title("Behavioral Shifts")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[3].grid(True)

        # Plot 5: Ratios and Distances
        w_dist = np.array(metrics["3-gram_wasserstein_raw"])
        max_w = 3 # max distance between two 3-grams
        axes[4].plot(x_deltas, w_dist / max_w, label="Norm. Wasserstein 3-gram", marker='x')
        
        overlap_arr = np.array(metrics["node_overlap"])
        visited_arr = np.array(metrics["total_nodes"][1:])
        axes[4].plot(x_deltas, overlap_arr / np.maximum(visited_arr, 1), label="Overlap / Visited Nodes", marker='x')
        
        perp_arr = np.array(metrics["state_visitation_perplexity"])
        axes[4].plot(checkpoint_labels, perp_arr / np.maximum(np.array(metrics["total_nodes"]), 1), label="Perplexity / Visited Nodes", marker='x')

        axes[4].set_xlabel("Checkpoint")
        axes[4].set_ylabel("Metrics (Scaled / Ratio)")
        axes[4].set_title("Normalized Distances and Ratios")
        axes[4].set_ylim(-0.05, 1.05)
        axes[4].set_xticks(checkpoint_labels)
        axes[4].tick_params(axis='x', rotation=45)
        axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[4].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = f"{args.output_prefix}_{run_name}_stacked_metrics.png"
        out_dir = os.path.dirname(plot_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if args.use_wandb:
            try:
                wandb.log({"plots/stacked_metrics": wandb.Image(fig)})
            except Exception as e:
                print(f"Wandb image logging error: {e}")
        plt.close(fig)
        print(f"Saved stacked metrics plot to {plot_path}")
    except Exception as e:
        print(f"Failed to generate the stacked metrics plot: {e}")

    # Save metrics to a file
    parsed_name = os.path.basename(os.path.normpath(args.data_dir))
    if not parsed_name:
        parsed_name = "metrics"
    
    out_dir = os.path.dirname(args.output_prefix) if os.path.dirname(args.output_prefix) else "metrics"
    metrics_file = os.path.join(out_dir, f"{parsed_name}_metrics.json")
    print(f"Saving metrics to {metrics_file}")
    
    # Convert numpy types to basic python types for JSON serialization
    serialized_metrics = {"checkpoints": checkpoint_labels}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serialized_metrics[k] = v.tolist()
        elif isinstance(v, list):
            serialized_metrics[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
        else:
            serialized_metrics[k] = v

    try:
        os.makedirs(os.path.dirname(os.path.abspath(metrics_file)), exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(serialized_metrics, f, indent=4)
    except Exception as e:
        print(f"Failed to save metrics to JSON: {e}")
            
    if args.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Wandb finish error: {e}")
    save_metrics_to_csv(metrics, "metrics.csv", args.wandb_run_name, "")
if __name__ == "__main__":
    main()

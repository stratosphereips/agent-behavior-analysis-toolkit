import argparse
import json
import os
from utils.data_utils import load_policies_from_directory
from utils.metrics import topological_shift, strategic_shift, traversal_depth, compute_entropy_metrics, calculate_temporal_action_entropy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils.plotting_utils import plot_behavioral_ontogeny
from utils.trajectory_utils import js_divergence_per_state, compute_trajectory_surprises

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
    
    if args.use_wandb:
        import wandb
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        else:
            run_name = f"cp_comp_{args.data_dir.replace('/', '_')}"
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
    policies = load_policies_from_directory(args.data_dir, args.max_trajectories, test_split=0.5)
    checkpoints = list(policies.keys())
    # Sort checkpoints by the integer value in the filename (assuming cp_XXXX format)
    # The dictionary keys might not be sorted correctly if they are strings like "cp_100", "cp_1000"
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else x)
    
    checkpoint_pairs = zip(checkpoints[:-1], checkpoints[1:])
    
    checkpoint_labels = [int(cp.split("_")[-1]) for cp in checkpoints]
    print(checkpoints)
    print(checkpoint_labels)

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

    metrics = {
        "topological_shift": [],
        "strategic_shift": [],
        "robust_traversal_depth": [],
        "robust_traversal_depth_noise": [],
        "reward": [],
        "reward_std": [],
        "effective_state_coverage": [],
        "effective_state_coverage_noise":[],
        "empirical_policy_certainty": [],
        "empirical_policy_certainty_noise": [],
        "unweighted_policy_certainty": [],
        "unweighted_policy_certainty_noise": [],
        "surprise_mean": [],
        "surprise_std": [],
        "temporal_action_entropy":[],
        "topological_shift_noise":[],
        "strategic_shift_noise":[],
        "traveral_depth_delta":[],
        "effective_stae_coverage_delta":[],
        "temporal_action_entropy_delta":[],
        "empirical_policy_certainty_delta":[],
        "unweighted_policy_certainty_delta":[],
        "reward_delta":[]
        
    }
    for i, timestep in enumerate(checkpoints):
        prev_lengths = {k: len(v) for k, v in metrics.items()}
        print(f"Processing checkpoint {timestep} (i={i})")
        # get policy and policy splits in current checkpoint
        current_policy = policies[timestep][0]
        current_policy_splitA = policies[timestep][2]
        current_policy_splitB = policies[timestep][3]
        # compute returns
        returns = current_policy.returns
        metrics["reward"].append(np.mean(returns))
        metrics["reward_std"].append(np.std(returns))

        # compute exploration volume and strategic confidence
        effective_state_coverage, empirical_policy_certainty, unweighted_h_pi = compute_entropy_metrics(current_policy._state_visitation_count, current_policy._state_action_map, len(GLOBAL_ACTIONS))
        metrics["effective_state_coverage"].append(effective_state_coverage)
        metrics["empirical_policy_certainty"].append(empirical_policy_certainty)
        metrics["unweighted_policy_certainty"].append(unweighted_h_pi)

        # compute exploration volume and strategic confidence error
        esc_A, epc_A, unweighted_h_pi_A = compute_entropy_metrics(current_policy_splitA._state_visitation_count, current_policy_splitA._state_action_map, len(GLOBAL_ACTIONS))
        esc_B, epc_B, unweighted_h_pi_B = compute_entropy_metrics(current_policy_splitB._state_visitation_count, current_policy_splitB._state_action_map, len(GLOBAL_ACTIONS))
        noise_effective_state_coverage = abs(esc_A - esc_B)
        noise_empirical_policy_certainty = abs(epc_A - epc_B)
        noise_unweighted_policy_certainty = abs(unweighted_h_pi_A - unweighted_h_pi_B)
        metrics["effective_state_coverage_noise"].append(noise_effective_state_coverage)
        metrics["empirical_policy_certainty_noise"].append(noise_empirical_policy_certainty)
        metrics["unweighted_policy_certainty_noise"].append(noise_unweighted_policy_certainty)

        # compute traversal depth
        # For Robust Traversal Depth, we want the Min of the splits, not the full dataset value?
        # The paper says: min(max(depth_A), max(depth_B)). 
        # But here we were computing it on the full policy too.
        # Let's stick to the paper definition for the main metric if we want to be strict,
        # OR we can keep tracking both.
        # The previous code tracked 'traversal_depth' (full) and 'traversal_depth_min' (robust).
        # We will map 'robust_traversal_depth' to the robust version (min of splits).
        
        traversal_depth_current_splitA = traversal_depth(current_policy_splitA.trajectories)
        traversal_depth_current_splitB = traversal_depth(current_policy_splitB.trajectories)
        metrics["robust_traversal_depth"].append(min(traversal_depth_current_splitA, traversal_depth_current_splitB))
        metrics["robust_traversal_depth_noise"].append(abs(traversal_depth_current_splitA - traversal_depth_current_splitB))

        metrics["temporal_action_entropy"].append(calculate_temporal_action_entropy(current_policy.trajectories, len(GLOBAL_ACTIONS)))

        # compute deltas of topological and strategic shift
        if i > 0:
            print(f"Computing deltas for checkpoint {checkpoints[i-1]} (i={i-1})")
            previous_policy = policies[checkpoints[i-1]][0]

            # compute noise values (from the A/B split)
            topo_shift_noise = topological_shift(current_policy_splitA._state_visitation_count, current_policy_splitB._state_visitation_count, noise_value=   0.0)
            strategic_shift_noise = strategic_shift(current_policy_splitA, current_policy_splitB, global_actions=GLOBAL_ACTIONS, noise_value=0.0)
            
            # compute deltas of topological and strategic shift
            metrics["topological_shift"].append(topological_shift(current_policy._state_visitation_count, previous_policy._state_visitation_count, noise_value=topo_shift_noise))
            metrics["strategic_shift"].append(strategic_shift(current_policy, previous_policy, global_actions=GLOBAL_ACTIONS, noise_value=strategic_shift_noise))
            metrics["topological_shift_noise"].append(topo_shift_noise)
            metrics["strategic_shift_noise"].append(strategic_shift_noise)
            # --- COMPUTE SURPRISE ---
            # 1. Compute per-state JS for normalization
            # GLOBAL_ACTIONS is a list, convert to set for js function if needed, or pass as is if function handles it.
            # js_divergence_per_state expects a set of actions
            js_div_per_state, _ = js_divergence_per_state(current_policy, previous_policy, set(GLOBAL_ACTIONS))
            
            # compute deltas of absolute metrics
            metrics["traveral_depth_delta"].append(metrics["robust_traversal_depth"][-1]-metrics["robust_traversal_depth"][-2])
            metrics["effective_stae_coverage_delta"].append(metrics["effective_state_coverage"][-1]-metrics["effective_state_coverage"][-2])
            metrics["temporal_action_entropy_delta"].append(metrics["temporal_action_entropy"][-1]-metrics["temporal_action_entropy"][-2])
            metrics["empirical_policy_certainty_delta"].append(metrics["empirical_policy_certainty"][-1]-metrics["empirical_policy_certainty"][-2])
            metrics["unweighted_policy_certainty_delta"].append(metrics["unweighted_policy_certainty"][-1]-metrics["unweighted_policy_certainty"][-2])
            metrics["reward_delta"].append(metrics["reward"][-1]-metrics["reward"][-2])
        if args.use_wandb:
            try:
                step_metrics = {"checkpoint": checkpoint_labels[i]}
                for k, v in metrics.items():
                    if len(v) > prev_lengths[k]:
                        step_metrics[k] = v[-1]
                wandb.log(step_metrics, step=checkpoint_labels[i])
            except Exception as e:
                print(f"Wandb logging error: {e}")
    
    print("\n--- Metric Correlations ---")
    corr_metrics = [
        "topological_shift", "strategic_shift", "robust_traversal_depth", 
        "reward", "effective_state_coverage", "empirical_policy_certainty", 
        "unweighted_policy_certainty", "temporal_action_entropy"
    ]
    valid_metrics = [m for m in corr_metrics if len(metrics.get(m, [])) > 0]
    
    if len(valid_metrics) > 1:
        min_len = min(len(metrics[m]) for m in valid_metrics)
        if min_len > 1:
            aligned_data = np.array([metrics[m][-min_len:] for m in valid_metrics])
            
            # Suppress warnings for constant data resulting in NaNs
            with np.errstate(invalid='ignore'):
                pearson_corr = np.corrcoef(aligned_data)
                spearman_corr, _ = stats.spearmanr(aligned_data, axis=1)
            
            print("Pearson Correlation:")
            header = f"{'':>28} " + " ".join([f"{m[:8]:>8}" for m in valid_metrics])
            print(header)
            for i, m1 in enumerate(valid_metrics):
                row_str = " ".join([f"{pearson_corr[i,j]:8.3f}" for j in range(len(valid_metrics))])
                print(f"{m1:>28} {row_str}")
                
            print("\nSpearman Correlation:")
            print(header)
            for i, m1 in enumerate(valid_metrics):
                row_str = " ".join([f"{spearman_corr[i,j]:8.3f}" for j in range(len(valid_metrics))])
                print(f"{m1:>28} {row_str}")

            if args.use_wandb:
                def log_corr_matrix(matrix, title, tag):
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        # Use np.nan_to_num to avoid matshow issues with pure NaN
                        clean_matrix = np.nan_to_num(matrix, nan=0.0)
                        cax = ax.matshow(clean_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                        fig.colorbar(cax)
                        
                        for i in range(len(valid_metrics)):
                            for j in range(len(valid_metrics)):
                                val = matrix[i,j]
                                text_val = f"{val:.2f}" if not np.isnan(val) else "NaN"
                                text_color = 'white' if abs(clean_matrix[i,j]) >= 0.5 else 'black'
                                ax.text(j, i, text_val, ha='center', va='center', color=text_color)
                                
                        ax.set_xticks(range(len(valid_metrics)))
                        ax.set_yticks(range(len(valid_metrics)))
                        ax.set_xticklabels(valid_metrics, rotation=45, ha='left')
                        ax.set_yticklabels(valid_metrics)
                        plt.title(title, pad=20)
                        plt.tight_layout()
                        wandb.log({tag: wandb.Image(fig)})
                        plt.close(fig)
                    except Exception as e:
                        print(f"Wandb {tag} logging error: {e}")
                        
                log_corr_matrix(pearson_corr, "Pearson Correlation", "correlations/pearson")
                log_corr_matrix(spearman_corr, "Spearman Correlation", "correlations/spearman")

    for nth in args.every_nth:
        print(f"Generating plot for every_nth={nth}")
        try:
            fig = plot_behavioral_ontogeny(checkpoint_labels, metrics, every_nth=nth)
            plt.savefig(f"{args.output_prefix}_every_{nth}.png", dpi=300)
            if args.use_wandb:
                try:
                    wandb.log({f"plots/every_{nth}": wandb.Image(fig)})
                except Exception as e:
                    print(f"Wandb image logging error: {e}")
            plt.close(fig)
        except Exception as e:
            print(f"Failed to generate plot for every_nth={nth}: {e}")

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

if __name__ == "__main__":
    main()

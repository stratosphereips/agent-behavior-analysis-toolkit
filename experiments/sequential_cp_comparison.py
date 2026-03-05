import argparse
from utils.data_utils import load_policies_from_directory
from utils.metrics import topological_shift, strategic_shift, traversal_depth, compute_entropy_metrics
import numpy as np
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
    
    args = parser.parse_args()
    
    if args.use_wandb:
        import wandb
        run_name = f"cp_comp_{args.data_dir.replace('/', '_')}"
        wandb.init(
            project="agent_trajectory_analysis",
            name=run_name,
            config=vars(args),
            tags=["sequential_cp_comparison"]
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
        "surprise_mean": [],
        "surprise_std": [],
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
        effective_state_coverage, empirical_policy_certainty = compute_entropy_metrics(current_policy._state_visitation_count, current_policy._state_action_map, len(GLOBAL_ACTIONS))
        metrics["effective_state_coverage"].append(effective_state_coverage)
        metrics["empirical_policy_certainty"].append(empirical_policy_certainty)

        # compute exploration volume and strategic confidence error
        esc_A, epc_A = compute_entropy_metrics(current_policy_splitA._state_visitation_count, current_policy_splitA._state_action_map, len(GLOBAL_ACTIONS))
        esc_B, epc_B = compute_entropy_metrics(current_policy_splitB._state_visitation_count, current_policy_splitB._state_action_map, len(GLOBAL_ACTIONS))
        noise_effective_state_coverage = abs(esc_A - esc_B)
        noise_empirical_policy_certainty = abs(epc_A - epc_B)
        metrics["effective_state_coverage_noise"].append(noise_effective_state_coverage)
        metrics["empirical_policy_certainty_noise"].append(noise_empirical_policy_certainty)

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
            
            # --- COMPUTE SURPRISE ---
            # 1. Compute per-state JS for normalization
            # GLOBAL_ACTIONS is a list, convert to set for js function if needed, or pass as is if function handles it.
            # js_divergence_per_state expects a set of actions
            js_div_per_state, _ = js_divergence_per_state(current_policy, previous_policy, set(GLOBAL_ACTIONS))
            
            # 2. Compute surprise for all trajectories in CURRENT policy relative to PREVIOUS
            all_surprises = []
            for t in current_policy.trajectories:
                surprises = compute_trajectory_surprises(t, current_policy, previous_policy, js_div_per_state)
                all_surprises.extend(surprises)
            
            if all_surprises:
                metrics["surprise_mean"].append(np.mean(all_surprises))
                metrics["surprise_std"].append(np.std(all_surprises))
            else:
                metrics["surprise_mean"].append(0.0)
                metrics["surprise_std"].append(0.0)
                
        else:
             # For the first checkpoint, no surprise relative to previous
             metrics["surprise_mean"].append(0.0)
             metrics["surprise_std"].append(0.0)

        if args.use_wandb:
            try:
                step_metrics = {"checkpoint": checkpoint_labels[i]}
                for k, v in metrics.items():
                    if len(v) > prev_lengths[k]:
                        step_metrics[k] = v[-1]
                wandb.log(step_metrics, step=checkpoint_labels[i])
            except Exception as e:
                print(f"Wandb logging error: {e}")
    
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
            
    if args.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Wandb finish error: {e}")

if __name__ == "__main__":
    main()

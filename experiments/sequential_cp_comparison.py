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
from typing import Any, Dict, List, Tuple
from utils.plotting_utils import plot_behavioral_ontogeny, plot_sequential_cp_metrics
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

    # r_true is optional: only present for trajectories recorded with the
    # true/underlying reward tracked in info["r_true"]. Older trajectories don't have it.
    r_true_returns = [rf for rf in (traj.total_r_true() for traj in policy.trajectories) if rf is not None]
    if r_true_returns:
        results["mean_r_true"] = float(np.mean(r_true_returns))
        results["std_r_true"] = float(np.std(r_true_returns))
    else:
        results["mean_r_true"] = None
        results["std_r_true"] = None
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

def policy_comparison_worker(policy1:EmpiricalPolicy, policy2:EmpiricalPolicy, ngram_cost_matrix:np.ndarray, global_ngrams:list[tuple], global_actions:list[int]):
    """
    Worker function for comparing two policies.
    Args:
        policy1: Current policy
        policy2: Previous policy
        ngram_cost_matrix: Cost matrix for n-grams
        global_ngrams: Global ordering of n-grams
        global_actions: Global ordering of actions
    Returns:
        dict: Dictionary of errors for the given checkpoint pair
    """
    results = {}
    topo_shift_values =  compute_decomposed_jsd(policy1._state_visitation_count, policy2._state_visitation_count)
    results["topological_shift"] = topo_shift_values["jsd_total"]
    results["topological_shift_overlap"] = topo_shift_values["jsd_overlap"]
    results["topological_shift_non_overlap"] = topo_shift_values["jsd_non_overlap"]

    # Directional split of the non-overlap term. compute_decomposed_jsd is called as
    # (current=policy1, previous=policy2), so p_A_unique is the probability mass the
    # CURRENT checkpoint places on newly visited states (frontier growth) and
    # p_B_unique is the mass the PREVIOUS checkpoint placed on now-abandoned states
    # (footprint collapse). On a single-checkpoint state the JSD integrand reduces to
    # exactly half that state's mass, so discovery + abandonment == jsd_non_overlap
    # (the split is exact, not an approximation). net = discovery - abandonment is the
    # signed flux: positive = net expansion, negative = net collapse.
    results["topological_shift_discovery"] = 0.5 * topo_shift_values["p_A_unique"]
    results["topological_shift_abandonment"] = 0.5 * topo_shift_values["p_B_unique"]
    results["topological_shift_net"] = (
        results["topological_shift_discovery"] - results["topological_shift_abandonment"]
    )

    results["strategic_shift"] = strategic_shift(policy1, policy2, global_actions=global_actions, noise_value=0.0)
    results["3-gram_wasserstein"] = compute_ngram_wasserstein_fast(policy1.trajectories, policy2.trajectories, global_ngrams, ngram_cost_matrix, n=3)
    results["node_overlap"] = len(set(policy1.states).intersection(set(policy2.states)))
    results["nodes_added"] = len(set(policy1.states).difference(set(policy2.states)))
    results["nodes_removed"] = len(set(policy2.states).difference(set(policy1.states)))
    return results

def estimate_noise_valus_for_policies(polcy1:EmpiricalPolicy, polcy2:EmpiricalPolicy, ngram_cost_matrix:np.ndarray, global_ngrams:list[tuple], global_actions:list[int], num_samples:int=100, percentile:float=0.95)->dict[str, Any]:
    # combine all trajectories into single pool
    all_trajectories = polcy1.trajectories + polcy2.trajectories
    N = len(all_trajectories)
    half_N = N // 2
    errors = {
        "topological_shift": [],
        "topological_shift_overlap": [],
        "topological_shift_non_overlap": [],
        "topological_shift_discovery": [],
        "topological_shift_abandonment": [],
        "topological_shift_net": [],
        "strategic_shift": [],
        "3-gram_wasserstein": [],
    }
    # M paired split-half resamples. Under the null the two halves are exchangeable,
    # so each split's metrics are a sampling-noise draw. Index i is aligned across all
    # metric lists (same split), which is what lets us form the joint max below.
    for i in range(num_samples):
        np.random.shuffle(all_trajectories)
        tmp_policy1 = EmpiricalPolicy(all_trajectories[:half_N], action_space=global_actions)
        tmp_policy2 = EmpiricalPolicy(all_trajectories[half_N:], action_space=global_actions)
        pairs = [(f"{i}_1", f"{i}_2")]
        pols = {f"{i}_1": tmp_policy1, f"{i}_2": tmp_policy2}
        tmp_errors = compare_checkpoints(pairs, pols, ngram_cost_matrix, global_ngrams, global_actions)
        for tmp_error in tmp_errors.values():
            for k in errors:
                errors[k].append(tmp_error[k])

    # Per-metric one-sided floors: kept for the report noise bands and for reading
    # WHICH metric drove a change. NaN-safe (strategic_shift is undefined when the
    # two halves share no states). 
    one_sided = [
        "topological_shift",
        "topological_shift_overlap",
        "topological_shift_non_overlap",
        "topological_shift_discovery",
        "topological_shift_abandonment",
        "strategic_shift",
        "3-gram_wasserstein",
    ]
    result = {}
    for k in one_sided:
        result[k] = float(np.nanquantile(errors[k], percentile))

    # Signed net frontier flux: two-sided band, centered ~0 under the null.
    alpha = 1.0 - percentile
    result["topological_shift_net_hi"] = float(np.nanquantile(errors["topological_shift_net"], 1.0 - alpha / 2.0))
    result["topological_shift_net_lo"] = float(np.nanquantile(errors["topological_shift_net"], alpha / 2.0))

    # Family-wise (max-statistic / Westfall-Young) calibration
    decision_metrics = ["topological_shift", "strategic_shift", "3-gram_wasserstein"]
    result["decision_metrics"] = decision_metrics
    result["null_mean"] = {m: float(np.nanmean(errors[m])) for m in decision_metrics}
    result["null_std"]  = {m: float(np.nanstd(errors[m]))  for m in decision_metrics}
    A  = np.array([errors[m] for m in decision_metrics], dtype=float)          # (3, M)
    mu = np.array([result["null_mean"][m] for m in decision_metrics])[:, None]
    sd = np.array([result["null_std"][m]  for m in decision_metrics])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (A - mu) / sd                                                      # (3, M)
    # A NaN (undefined) or zero-variance metric cannot be a split's max -> drop it.
    Z[~np.isfinite(Z)] = -np.inf
    zmax = np.max(Z, axis=0)                                                   # (M,)
    zmax = zmax[np.isfinite(zmax)]

    if zmax.size:
        result["zmax_p90"] = float(np.quantile(zmax, 0.90))
        result["zmax_p95"] = float(np.quantile(zmax, 0.95))
        result["zmax_p99"] = float(np.quantile(zmax, 0.99))
    else:
        result["zmax_p90"] = result["zmax_p95"] = result["zmax_p99"] = float("inf")
    return result

def estimate_noise_sensitivity_for_policies(
    policy1: EmpiricalPolicy,
    policy2: EmpiricalPolicy,
    ngram_cost_matrix: np.ndarray,
    global_ngrams: list[tuple],
    global_actions: list[int],
    m_values: list[int],
    n_subsamples: int = 20,
    percentile: float = 0.95,
) -> dict:
    """
    Estimate noise threshold sensitivity to bootstrap sample count M.

    Runs the full bootstrap once at max(m_values), saves all raw metric values,
    then for each M in m_values subsamples M values n_subsamples times and
    computes the percentile threshold each time. Returns mean and std of those
    estimates, normalized by the max-M reference.

    Returns:
        dict keyed by metric name, each value is a dict:
            {M: {"mean": float, "std": float, "mean_normalized": float, "std_normalized": float}}
    """
    m_max = max(m_values)

    # --- collect m_max raw bootstrap values ---
    all_trajectories = policy1.trajectories + policy2.trajectories
    half_N = len(all_trajectories) // 2

    all_returns = policy1.returns + policy2.returns

    # Note: discovery/abandonment are tracked here (non-negative, one-sided like the
    # other metrics). The signed net flux is omitted from this sweep because its
    # threshold is two-sided and zero-centered, so the one-sided percentile ratio
    # used below is not the right stability summary for it.
    raw: dict[str, list[float]] = {
        "topological_shift": [],
        "topological_shift_overlap": [],
        "topological_shift_non_overlap": [],
        "topological_shift_discovery": [],
        "topological_shift_abandonment": [],
        "strategic_shift": [],
        "3-gram_wasserstein": [],
        "mean_return_diff": [],
    }

    for i in range(m_max):
        shuffled = all_trajectories[:]
        np.random.shuffle(shuffled)
        tmp_p1 = EmpiricalPolicy(shuffled[:half_N], action_space=global_actions)
        tmp_p2 = EmpiricalPolicy(shuffled[half_N:], action_space=global_actions)
        tmp_pairs = [(f"{i}_1", f"{i}_2")]
        tmp_policies = {f"{i}_1": tmp_p1, f"{i}_2": tmp_p2}
        tmp_errors = compare_checkpoints(tmp_pairs, tmp_policies, ngram_cost_matrix, global_ngrams, global_actions)
        for err in tmp_errors.values():
            raw["topological_shift"].append(err["topological_shift"])
            raw["topological_shift_overlap"].append(err["topological_shift_overlap"])
            raw["topological_shift_non_overlap"].append(err["topological_shift_non_overlap"])
            raw["topological_shift_discovery"].append(err["topological_shift_discovery"])
            raw["topological_shift_abandonment"].append(err["topological_shift_abandonment"])
            raw["strategic_shift"].append(err["strategic_shift"])
            raw["3-gram_wasserstein"].append(err["3-gram_wasserstein"])

        # reward noise: split pooled returns 50/50, measure |mean difference|
        shuffled_returns = all_returns[:]
        np.random.shuffle(shuffled_returns)
        half_R = len(shuffled_returns) // 2
        raw["mean_return_diff"].append(
            abs(np.mean(shuffled_returns[:half_R]) - np.mean(shuffled_returns[half_R:]))
        )

    # reference threshold at m_max
    ref = {k: np.quantile(v, percentile) for k, v in raw.items()}

    # --- subsample for each M ---
    results: dict[str, dict[int, dict]] = {k: {} for k in raw}

    for M in m_values:
        for metric, values in raw.items():
            subsample_quantiles = [
                np.quantile(np.random.choice(values, size=M, replace=False), percentile)
                for _ in range(n_subsamples)
            ]
            mean_q = float(np.mean(subsample_quantiles))
            std_q = float(np.std(subsample_quantiles))
            ref_val = ref[metric] if ref[metric] > 0 else 1.0
            results[metric][M] = {
                "mean": mean_q,
                "std": std_q,
                "mean_normalized": mean_q / ref_val,
                "std_normalized": std_q / ref_val,
            }

    return results


def estimate_noise_sensitivity(
    policy_pairs: list[tuple],
    policies: dict,
    ngram_cost_matrix: np.ndarray,
    global_ngrams: list[tuple],
    global_actions: list[int],
    m_values: list[int],
    n_subsamples: int = 20,
    percentile: float = 0.95,
) -> dict:
    """
    Run estimate_noise_sensitivity_for_policies for all checkpoint pairs in parallel.

    Returns:
        dict keyed by (p1, p2), each value is the output of
        estimate_noise_sensitivity_for_policies — i.e.
        {metric: {M: {"mean", "std", "mean_normalized", "std_normalized"}}}
    """
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                estimate_noise_sensitivity_for_policies,
                policies[p1], policies[p2],
                ngram_cost_matrix, global_ngrams, global_actions,
                m_values, n_subsamples, percentile,
            ): (p1, p2)
            for p1, p2 in policy_pairs
        }
        for f in as_completed(futures):
            p1, p2 = futures[f]
            try:
                results[(p1, p2)] = f.result()
            except Exception as e:
                print(f"Error estimating noise sensitivity for ({p1}, {p2}): {e}")
    return results


def estimate_noise_values(policy_pairs, policies, ngram_cost_matrix, global_ngrams, global_actions, num_samples:int=50, percentile:float=0.95):
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(estimate_noise_valus_for_policies, policies[p1], policies[p2], ngram_cost_matrix, global_ngrams, global_actions, num_samples, percentile): (p1, p2)
            for p1, p2 in policy_pairs
        }
        for f in as_completed(futures):
            p1, p2 = futures[f]
            try:
                results[(p1, p2)] = f.result()
            except Exception as e:
                print(f"Error estimating noise values for policy pair: {e}")
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
            executor.submit(policy_comparison_worker, checkpoint_policies[current], checkpoint_policies[previous], ngram_cost_matrix, global_ngrams, global_actions): (current, previous)
            for previous, current in checkpoint_pairs
        }
        for f in as_completed(futures):
            current, previous = futures[f]
            try:
                results[(current, previous)] = f.result()
            except Exception as e:
                print(f"Error comparing checkpoint {current} and {previous}: {e}")
    return results

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
    parser.add_argument("--output_prefix", type=str, default="figures/behavioral_ontology", help="Prefix for output image")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store all output files (overrides --output_prefix directory)")
    parser.add_argument("--noise_num_samples", type=int, default=20, help="Number of samples for noise estimation")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable Weights & Biases logging")
    parser.add_argument("--use_wanndb", action="store_true", dest="use_wandb", help=argparse.SUPPRESS) # alias
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="Tags for Weights & Biases run")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Run name for Weights & Biases")
    parser.add_argument("--env", type=str, default="default", help="Environment type (e.g., default, netsecgame)")

    
    args = parser.parse_args()
    
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    else:
        path_parts = os.path.normpath(args.data_dir).split(os.sep)
        try:
            if 'behavioral_ontogeny' in path_parts:
                idx = path_parts.index('behavioral_ontogeny')
                env_name = path_parts[idx+1]
                model_name = path_parts[idx+2]
                mode_name = path_parts[idx+3]
                run_name = f"{env_name}_{model_name}_{mode_name}"
            else:
                run_name = f"cp_comp_{args.data_dir.replace('/', '_')}"
        except IndexError:
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

    action_encoder = None
    state_encoder = None
    if args.env in ["netsecgame", "aidojo"]:
        from utils.aidojo_utils import aidojo_state_str_from_dict, aidojo_action_type_from_dict
        action_encoder = aidojo_action_type_from_dict
        state_encoder = aidojo_state_str_from_dict

    # load empirical policies from files
    policies = load_policies_from_directory(
        args.data_dir, 
        args.max_trajectories, 
        action_encoder=action_encoder,
        state_encoder=state_encoder,
        test_split=None
    )
    checkpoints = list(policies.keys())
    # Sort checkpoints by the integer value in the filename (assuming cp_XXXX format)
    # The dictionary keys might not be sorted correctly if they are strings like "cp_100", "cp_1000"
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]) if "_" in x and x.split("_")[-1].isdigit() else x)
    

    checkpoint_pairs = list(zip(checkpoints[:-1], checkpoints[1:]))
    #checkpoint_pairs = [(checkpoints[0], cp) for cp in checkpoints[1:]]
    
    checkpoint_labels = [int(cp.split("_")[-1].lstrip("ep")) for cp in checkpoints]

    checkpoint_policies = {}
    for cp_key in checkpoints:
        checkpoint_policies[cp_key] = policies[cp_key][0]

    
    # Determine GLOBAL_ACTIONS
    if args.env in ["netsecgame", "aidojo"]:
        from netsecgame.game_components import ActionType
        GLOBAL_ACTIONS = sorted([a for a in list(ActionType) if a not in ['ActionType.JoinGame', 'ActionType.QuitGame', 'ActionType.ResetGame']], key=lambda x: str(x))
        print(f"Using netsecgame ActionType for GLOBAL_ACTIONS: {GLOBAL_ACTIONS}")
    elif args.num_actions is not None:
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
            GLOBAL_ACTIONS = sorted(list(all_actions), key=lambda x: str(x))
            print(f"Inferred GLOBAL_ACTIONS from trajectories: {GLOBAL_ACTIONS}")
    else:
        GLOBAL_ACTIONS = [0, 1] # Fallback default

    # precompute ngrams and distance matrix
    global_ngrams_3, cost_matrix_3 = build_global_environment_cache(GLOBAL_ACTIONS, n=3)

    checkpoint_stats = compute_checkpoint_stats(checkpoint_policies)
    
    # compute errors for each checkpoint
    #errors = compute_errors_per_checkpoint(checkpoint_policies, cost_matrix_3, global_ngrams_3, GLOBAL_ACTIONS)

    # estimate noise values for each checkpoint
    noise_values = estimate_noise_values(checkpoint_pairs, checkpoint_policies, cost_matrix_3, global_ngrams_3, GLOBAL_ACTIONS, num_samples=args.noise_num_samples)
    
    # compare checkpoints
    comparisons = compare_checkpoints(checkpoint_pairs, checkpoint_policies, cost_matrix_3, global_ngrams_3, GLOBAL_ACTIONS)

    # merge results
    metrics = { 
        "topological_shift_raw": [],
        "mean_return": [],
        "std_return": [],
        "mean_r_true": [],
        "std_r_true": [],
        "state_visitation_perplexity": [],
        "topological_shift_overlap_raw": [],
        "topological_shift_non_overlap_raw": [],
        # directional decomposition of the non-overlap term
        "topological_shift_discovery_raw": [],
        "topological_shift_abandonment_raw": [],
        "topological_shift_net_raw": [],
        "strategic_shift_raw": [],
        "3-gram_wasserstein_raw": [],
        "topological_shift_noise_threshold": [],
        "topological_shift_overlap_noise_threshold": [],
        "topological_shift_non_overlap_noise_threshold": [],
        # floors for the directional components (net is two-sided)
        "topological_shift_discovery_noise_threshold": [],
        "topological_shift_abandonment_noise_threshold": [],
        "topological_shift_net_noise_hi": [],
        "topological_shift_net_noise_lo": [],
        "strategic_shift_noise_threshold": [],
        "3-gram_wasserstein_noise_threshold": [],
        # family-wise (max-statistic) calibration across the decision metrics
        "zmax_p90": [],
        "zmax_p95": [],
        "zmax_p99": [],
        # per-metric null mean/std used to z-score the raw values against zmax_p95
        "null_mean_topological_shift": [],
        "null_std_topological_shift": [],
        "null_mean_strategic_shift": [],
        "null_std_strategic_shift": [],
        "null_mean_3-gram_wasserstein": [],
        "null_std_3-gram_wasserstein": [],
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
        metrics["mean_r_true"].append(checkpoint_stats[timestep]["mean_r_true"])
        metrics["std_r_true"].append(checkpoint_stats[timestep]["std_r_true"])
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
            metrics["topological_shift_discovery_raw"].append(comparisons[current_ts, prev_ts]["topological_shift_discovery"])
            metrics["topological_shift_abandonment_raw"].append(comparisons[current_ts, prev_ts]["topological_shift_abandonment"])
            metrics["topological_shift_net_raw"].append(comparisons[current_ts, prev_ts]["topological_shift_net"])
            metrics["strategic_shift_raw"].append(comparisons[current_ts, prev_ts]["strategic_shift"])
            metrics["3-gram_wasserstein_raw"].append(comparisons[current_ts, prev_ts]["3-gram_wasserstein"])
            metrics["node_overlap"].append(comparisons[current_ts, prev_ts]["node_overlap"])
            metrics["nodes_added"].append(comparisons[current_ts, prev_ts]["nodes_added"])
            metrics["nodes_removed"].append(comparisons[current_ts, prev_ts]["nodes_removed"])

            # add noise threshold deltas
            metrics["topological_shift_noise_threshold"].append(noise_values[prev_ts, current_ts]["topological_shift"])
            metrics["topological_shift_overlap_noise_threshold"].append(noise_values[prev_ts, current_ts]["topological_shift_overlap"])
            metrics["topological_shift_non_overlap_noise_threshold"].append(noise_values[prev_ts, current_ts]["topological_shift_non_overlap"])
            metrics["topological_shift_discovery_noise_threshold"].append(noise_values[prev_ts, current_ts]["topological_shift_discovery"])
            metrics["topological_shift_abandonment_noise_threshold"].append(noise_values[prev_ts, current_ts]["topological_shift_abandonment"])
            metrics["topological_shift_net_noise_hi"].append(noise_values[prev_ts, current_ts]["topological_shift_net_hi"])
            metrics["topological_shift_net_noise_lo"].append(noise_values[prev_ts, current_ts]["topological_shift_net_lo"])
            metrics["strategic_shift_noise_threshold"].append(noise_values[prev_ts, current_ts]["strategic_shift"])
            metrics["3-gram_wasserstein_noise_threshold"].append(noise_values[prev_ts, current_ts]["3-gram_wasserstein"])
            metrics["zmax_p90"].append(noise_values[prev_ts, current_ts]["zmax_p90"])
            metrics["zmax_p95"].append(noise_values[prev_ts, current_ts]["zmax_p95"])
            metrics["zmax_p99"].append(noise_values[prev_ts, current_ts]["zmax_p99"])
            for m in ("topological_shift", "strategic_shift", "3-gram_wasserstein"):
                metrics[f"null_mean_{m}"].append(noise_values[prev_ts, current_ts]["null_mean"][m])
                metrics[f"null_std_{m}"].append(noise_values[prev_ts, current_ts]["null_std"][m])

        if args.use_wandb:
            try:
                step_metrics = {"checkpoint": checkpoint_labels[i]}
                for k, v in metrics.items():
                    if len(v) > 0 and v[-1] is not None:
                        step_metrics[k] = v[-1]
                wandb.log(step_metrics, step=checkpoint_labels[i])
            except Exception as e:
                print(f"Wandb logging error: {e}")
    
    # Visualization of experiments
    try:
        fig = plot_sequential_cp_metrics(checkpoint_labels, metrics, run_name)
        if args.output_dir:
            plot_path = os.path.join(args.output_dir, f"{run_name}_stacked_metrics.png")
        else:
            plot_path = f"{args.output_prefix}_{run_name}_stacked_metrics.png"
        plot_name = os.path.basename(plot_path)
        if len(plot_name) > 250:
            import hashlib
            hash_str = hashlib.md5(plot_name.encode()).hexdigest()[:8]
            suffix = "_stacked_metrics.png"
            max_len = 250 - len(suffix) - 1 - len(hash_str)
            plot_name = plot_name[:max_len] + "_" + hash_str + suffix
            plot_path = os.path.join(os.path.dirname(plot_path), plot_name)

        out_dir = os.path.dirname(plot_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        # Calculate DPI to ensure image does not exceed 8192x4096
        fig_width, fig_height = fig.get_size_inches()
        max_dpi_w = 8192 / fig_width
        max_dpi_h = 4096 / fig_height
        target_dpi = min(300, max_dpi_w, max_dpi_h)

        fig.savefig(plot_path, dpi=target_dpi, bbox_inches='tight')

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
    
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(args.output_prefix) if os.path.dirname(args.output_prefix) else "metrics"
    metrics_file = os.path.join(out_dir, f"{parsed_name}_{args.noise_num_samples}_metrics.json")
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
    #save_metrics_to_csv(metrics, "metrics.csv", args.wandb_run_name, "")
if __name__ == "__main__":
    main()

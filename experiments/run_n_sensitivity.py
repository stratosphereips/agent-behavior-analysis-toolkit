"""
N sensitivity experiment — metric stability vs number of evaluation trajectories.

For each checkpoint k, subsamples N trajectories (without replacement) from the
full pool and computes all metrics against the full previous checkpoint (k-1).
Repeats n_subsamples times to estimate variance. Reports CV (std/mean) across
subsamples for each metric at each N.

The key comparison: behavioral metrics (Seff, ΔTopo, ΔStrat, W3) vs mean return.
If behavioral metrics reach low CV at smaller N than reward, this supports the
claim that structural profiling requires fewer evaluation rollouts than reward
monitoring — a practical advantage in settings where trajectory collection is
costly or constrained.

Usage:
    python experiments/run_n_sensitivity.py \\
        --data_dir <path/to/trajectories> \\
        --output <path/to/results.json> \\
        --num_actions 4 \\
        --n_subsamples 20
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.data_utils import load_policies_from_directory
from utils.metrics import compute_perplexity_from_counts
from experiments.sequential_cp_comparison import (
    build_global_environment_cache,
    policy_comparison_worker,
)
from trajectory import EmpiricalPolicy

# Default N grid. Keep N at most ~40% of the collected pool size: subsampling
# WITHOUT replacement deflates variance by sqrt(1 - N/pool), so points with N
# close to the pool look spuriously stable (at N = pool size, std is exactly 0).
# With a 500-trajectory pool the honest ceiling is N~200; a 1000-pool allows ~400.
N_VALUES = [10, 50, 100, 200]
N_SUBSAMPLES = 20


def evaluate_at_n(
    trajectories_k: list,
    policy_prev: EmpiricalPolicy,
    ngram_cost_matrix,
    global_ngrams: list,
    global_actions: list,
    N: int,
    n_subsamples: int,
    replace: bool = False,
) -> dict:
    """
    Resample N trajectories from checkpoint k n_subsamples times.
    For each resample compute all metrics against the full policy_prev.
    Returns mean, std, and CV for each metric at this N.

    replace=False  -> subsampling without replacement. Estimates the variance of
                      an N-trajectory estimate, but is only valid for N well below
                      the pool size (variance is deflated by sqrt(1 - N/pool) and
                      is exactly 0 at N = pool size).
    replace=True   -> bootstrap with replacement. Gives a non-degenerate variance
                      estimate even at N = pool size, so it is used to measure the
                      precision of the estimate at the full operating budget.
    """
    if N > len(trajectories_k) and not replace:
        N = len(trajectories_k)

    metric_samples: dict[str, list[float]] = {
        "seff": [],
        "topological_shift": [],
        "topological_shift_overlap": [],
        "topological_shift_non_overlap": [],
        "strategic_shift": [],
        "3-gram_wasserstein": [],
        "mean_return": [],
    }

    for _ in range(n_subsamples):
        indices = np.random.choice(len(trajectories_k), size=N, replace=replace)
        subsample = [trajectories_k[i] for i in indices]

        policy_sub = EmpiricalPolicy(subsample, action_space=global_actions)

        # Seff
        seff = compute_perplexity_from_counts(policy_sub._state_visitation_count)
        metric_samples["seff"].append(seff)

        # mean return
        metric_samples["mean_return"].append(np.mean(policy_sub.returns))

        # comparative metrics vs full previous checkpoint
        cmp = policy_comparison_worker(
            policy_sub, policy_prev, ngram_cost_matrix, global_ngrams, global_actions
        )
        metric_samples["topological_shift"].append(cmp["topological_shift"])
        metric_samples["topological_shift_overlap"].append(cmp["topological_shift_overlap"])
        metric_samples["topological_shift_non_overlap"].append(cmp["topological_shift_non_overlap"])
        metric_samples["strategic_shift"].append(cmp["strategic_shift"])
        metric_samples["3-gram_wasserstein"].append(cmp["3-gram_wasserstein"])

    out = {}
    for metric, values in metric_samples.items():
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = float(std / mean) if mean > 0 else float("nan")
        out[metric] = {"mean": mean, "std": std, "cv": cv}
    return out


def process_pair(args):
    cp_prev, cp_curr, trajectories_prev, trajectories_curr, \
        ngram_cost_matrix, global_ngrams, global_actions, n_subsamples, n_values = args

    policy_prev = EmpiricalPolicy(trajectories_prev, action_space=global_actions)

    result = {}
    for N in n_values:
        result[N] = evaluate_at_n(
            trajectories_curr, policy_prev,
            ngram_cost_matrix, global_ngrams, global_actions,
            N, n_subsamples,
        )

    # Operating-budget point: precision of the estimate computed from the FULL pool,
    # measured by bootstrap (with replacement) so it is non-degenerate at N = pool
    # size. This is the directly-measured precision at the budget actually used,
    # not an extrapolation from the subsampling curve.
    op_N = len(trajectories_curr)
    op_stats = evaluate_at_n(
        trajectories_curr, policy_prev,
        ngram_cost_matrix, global_ngrams, global_actions,
        op_N, n_subsamples, replace=True,
    )
    return cp_prev, cp_curr, result, op_N, op_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_actions", type=int, default=None)
    parser.add_argument("--n_subsamples", type=int, default=N_SUBSAMPLES)
    parser.add_argument("--n_values", type=int, nargs="+", default=N_VALUES,
                        help="Subsample sizes to probe. Keep each at most ~40%% of "
                             "the collected pool (e.g. <=200 for a 500-traj pool, "
                             "<=400 for a 1000-traj pool) to avoid finite-population "
                             "variance deflation.")
    args = parser.parse_args()
    n_values = sorted(args.n_values)

    policies = load_policies_from_directory(args.data_dir, max_trajectories=1000)
    checkpoints = sorted(
        policies.keys(),
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
    )
    checkpoint_pairs = list(zip(checkpoints[:-1], checkpoints[1:]))

    if args.num_actions is not None:
        global_actions = list(range(args.num_actions))
    else:
        all_actions = set()
        for cp in checkpoints:
            all_actions.update(policies[cp][0].actions)
        global_actions = sorted(all_actions, key=str)
    print(f"Action space: {global_actions}")

    global_ngrams, cost_matrix = build_global_environment_cache(global_actions, n=3)

    # Warn if any requested N is too large a fraction of the available pool.
    min_pool = min(len(policies[cp][0].trajectories) for cp in checkpoints)
    if max(n_values) > 0.4 * min_pool:
        print(f"WARNING: largest N={max(n_values)} exceeds 40% of the smallest pool "
              f"({min_pool}); finite-population deflation will make large-N points "
              f"look spuriously stable. Consider N <= {int(0.4 * min_pool)}.")

    print(f"Running N sensitivity over {len(checkpoint_pairs)} checkpoint pairs, "
          f"N={n_values}, n_subsamples={args.n_subsamples} ...")

    work = [
        (
            cp_prev, cp_curr,
            policies[cp_prev][0].trajectories,
            policies[cp_curr][0].trajectories,
            cost_matrix, global_ngrams, global_actions,
            args.n_subsamples, n_values,
        )
        for cp_prev, cp_curr in checkpoint_pairs
    ]

    sensitivity = {}
    operating = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pair, w): w for w in work}
        for f in as_completed(futures):
            try:
                cp_prev, cp_curr, result, op_N, op_stats = f.result()
                key = f"{cp_prev}__{cp_curr}"
                sensitivity[key] = {str(N): result[N] for N in n_values}
                operating[key] = {"N": op_N, "stats": op_stats}
            except Exception as e:
                print(f"Error: {e}")

    # --- Post-process: add a range-normalized std for cross-metric comparison ---
    # CV (std/mean) is unstable wherever a metric's mean is near or crosses zero:
    #   - mean_return crosses zero in Taxi/MountainCar (CV blows up / flips sign),
    #   - the shift metrics decay to ~0 at convergence (CV blows up).
    # We therefore also report std normalized by each metric's *signal range* across
    # the checkpoint series (max - min of the per-pair means at the largest N). This
    # is robust to both pathologies and directly comparable across metrics with
    # different units/scales: it answers "how big is the sampling noise at N relative
    # to how much this metric actually moves over training?". CV is retained for the
    # bounded behavioral metrics, where it remains meaningful during the active phase.
    ref_N = str(max(n_values))
    metric_names = [
        "seff",
        "topological_shift",
        "topological_shift_overlap",
        "topological_shift_non_overlap",
        "strategic_shift",
        "3-gram_wasserstein",
        "mean_return",
    ]
    signal_range = {}
    for metric in metric_names:
        means = [
            sensitivity[k][ref_N][metric]["mean"]
            for k in sensitivity
            if ref_N in sensitivity[k]
            and metric in sensitivity[k][ref_N]
            and np.isfinite(sensitivity[k][ref_N][metric]["mean"])
        ]
        rng = (max(means) - min(means)) if len(means) >= 2 else float("nan")
        signal_range[metric] = float(rng) if (np.isfinite(rng) and rng > 0) else float("nan")

    def add_std_over_range(stats_by_metric):
        for metric, stats in stats_by_metric.items():
            rng = signal_range.get(metric, float("nan"))
            stats["std_over_range"] = (
                float(stats["std"] / rng)
                if (np.isfinite(rng) and rng > 0)
                else float("nan")
            )

    for k in sensitivity:
        for N_key in sensitivity[k]:
            add_std_over_range(sensitivity[k][N_key])

    # Same range-normalization for the operating-budget (full-pool bootstrap) point,
    # so it lands on the same axis/tolerance as the subsampling curve.
    for k in operating:
        add_std_over_range(operating[k]["stats"])

    output = {
        "n_subsamples": args.n_subsamples,
        "n_values": n_values,
        "signal_range": signal_range,
        "pairs": sensitivity,
        "operating": operating,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")
    print("Signal range (denominator for std_over_range) per metric:")
    for m, r in signal_range.items():
        print(f"  {m:34s} {r}")

    # Operating-budget precision: mean std_over_range across pairs at the full pool.
    if operating:
        op_Ns = [operating[k]["N"] for k in operating]
        print(f"\nOperating-budget precision (full-pool bootstrap, "
              f"N~{int(np.median(op_Ns))}), mean std_over_range across pairs:")
        for m in metric_names:
            vals = [
                operating[k]["stats"][m]["std_over_range"]
                for k in operating
                if m in operating[k]["stats"]
                and np.isfinite(operating[k]["stats"][m]["std_over_range"])
            ]
            if vals:
                print(f"  {m:34s} {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()

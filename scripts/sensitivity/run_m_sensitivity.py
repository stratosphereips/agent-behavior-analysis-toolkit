"""
M sensitivity experiment — noise threshold stability vs bootstrap sample count.

Loads trajectory data from a given directory, runs estimate_noise_sensitivity
across all consecutive checkpoint pairs, and saves results to JSON.

Usage:
    python -m scripts.sensitivity.run_m_sensitivity \
        --data_dir <path/to/trajectories> \
        --output <path/to/results.json> \
        --num_actions 4 \
        --n_subsamples 20
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.data_utils import load_policies_from_directory
from scripts.behavioral_fingerprint.sequential_cp_comparison import (
    build_global_environment_cache,
    estimate_noise_sensitivity,
)

M_VALUES = [1, 2, 5, 10, 20, 50, 100, 200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_actions", type=int, default=None)
    parser.add_argument("--n_subsamples", type=int, default=20,
                        help="Number of subsamples per M value per checkpoint pair")
    parser.add_argument("--percentile", type=float, default=0.95)
    args = parser.parse_args()

    policies = load_policies_from_directory(args.data_dir, max_trajectories=1000)
    checkpoints = sorted(policies.keys(),
                         key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
    checkpoint_policies = {cp: policies[cp][0] for cp in checkpoints}
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

    print(f"Running M sensitivity over {len(checkpoint_pairs)} checkpoint pairs, "
          f"M={M_VALUES}, n_subsamples={args.n_subsamples} ...")

    sensitivity = estimate_noise_sensitivity(
        checkpoint_pairs, checkpoint_policies,
        cost_matrix, global_ngrams, global_actions,
        m_values=M_VALUES,
        n_subsamples=args.n_subsamples,
        percentile=args.percentile,
    )

    # serialise: keys are tuples, convert to strings
    serialisable = {}
    for (p1, p2), metric_data in sensitivity.items():
        key = f"{p1}__{p2}"
        serialisable[key] = {}
        for metric, m_dict in metric_data.items():
            serialisable[key][metric] = {str(M): vals for M, vals in m_dict.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

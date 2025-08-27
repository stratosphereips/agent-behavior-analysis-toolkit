from utils.trajectory_utils import load_trajectories_from_json
from utils.trajectory_utils import empirical_policy_statistics
from utils.trajectory_utils import find_trajectory_segments, cluster_segments
from utils.plotting_utils   import plot_segment_cluster_features, plot_trajectory_network_colored_nodes_by_cluster
from utils.plotting_utils   import plot_trajectory_segments, plot_action_per_step_distribution, plot_trajectory_heatmap
from utils.plotting_utils   import plot_trajectory_heatmap, visualize_clusters
from utils.trajectory_utils import get_trajectory_action_change
from utils.trajectory_utils import get_clusters_per_step
from utils.trajectory_utils import compute_trajectory_surprises,compute_lambda_returns
from trajectory import EmpiricalPolicy
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor


def create_empirical_policy(json_file, checkpoint_id):
    """
    Create an empirical policy from the trajectories in a JSON file.
    """
    trajectories, trajectory_metadata = load_trajectories_from_json(
        json_file, load_metadata=True, max_trajectories=50
    )
    empirical_policy = EmpiricalPolicy(trajectories)
    return checkpoint_id, trajectories, trajectory_metadata, empirical_policy

def process_single_trajectory(args):
        """
        Process a single trajectory, finds segments and extract features.
        """
        traj_idx, t, curr_policy, prev_policy, checkpoint_id = args
        rewards = np.array(t.rewards)
        surprises = np.array(compute_trajectory_surprises(t, curr_policy, prev_policy, epsilon=1e-12))
        lambda_returns = np.array(compute_lambda_returns(rewards))
        return find_trajectory_segments(
            surprises=surprises,
            rewards=rewards,
            lambda_returns=lambda_returns,
            trajectory_id=f"{checkpoint_id}_{traj_idx}"
        )

def process_comparison(checkpoint_id, trajectories, metadata, prev_policy, curr_policy):
    """
    Worker function: compares two checkpoints (i-1, i) and computes log_data.
    Returns raw metrics + images as bytes (safe to pass across processes).
    """
    log_data = {"static_graph_metrics": empirical_policy_statistics(curr_policy)}
    log_data["Cluster Feature Summary"] = None
    log_data["Segment Surprise Plot"] = None
    log_data['segmentation_metrics'] = {
        "segments": 0,
        "unique_segments": 0,
        "clusters": 0,
        "mean_segment_in_cluster": 0.0,
        "mean_unique_segment_in_cluster": 0.0,
        "unique_trajectories": len(set(trajectories))
    }

    # find segments - run in parallel
    segments = []
    with ThreadPoolExecutor() as pool:
        results = pool.map(
            process_single_trajectory,
            ((traj_idx, t, curr_policy, prev_policy, checkpoint_id) for traj_idx, t in enumerate(trajectories))
        )
        for segs in results:
            segments += segs

    if segments:
        log_data['segmentation_metrics'].update({
            "segments": len(segments),
            "unique_segments": len({s["features"] for s in segments})
        })

        clustering = cluster_segments(segments)
        segments_per_cluster = [len(segs) for segs in clustering.values()]
        unique_segments_per_cluster = [
            len({seg["features"] for seg in segs})
            for segs in clustering.values()
        ]

        log_data['segmentation_metrics'].update({
            "clusters": len(clustering),
            "mean_segment_in_cluster": np.mean(segments_per_cluster) if segments_per_cluster else 0.0,
            "mean_unique_segment_in_cluster": np.mean(unique_segments_per_cluster) if unique_segments_per_cluster else 0.0
        })

        # --- make plots, but save them as PNG bytes (wandb.Image is only in main proc) ---
        figs = {}

        fig = plot_segment_cluster_features(clustering)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Cluster Feature Summary"] = buf.read()
        plt.close(fig)

        fig = plot_trajectory_segments(trajectories, curr_policy, prev_policy, f"Surprise_plot_cp_{checkpoint_id}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Segment Surprise Plot"] = buf.read()
        plt.close(fig)

        num_actions = 6  # curr_policy.num_actions
        fig = plot_action_per_step_distribution(trajectories, num_actions, normalize=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Action Distribution Plot"] = buf.read()
        plt.close(fig)

        unique_trajectories = set(trajectories)
        max_len = max(len(t) for t in unique_trajectories)
        fig = visualize_clusters(clustering, max_len)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Cluster Visualization"] = buf.read()
        plt.close(fig)

        log_data["_figs"] = figs  # stash for main process

    return checkpoint_id, log_data, metadata

class TrajectoryReplay:
    """
    Class for loading and processing of recorded trajectories
    """

    def __init__(self, trajectory_dir, **kwargs):
        self.trajectories = []
        self.json_files = sorted([os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir) if f.endswith(".json") or f.endswith(".jsonl")])
        print(f"Found {len(self.json_files)} JSON files in {trajectory_dir}")
        self._previous_policy = None
        wandb_project = kwargs.get("wandb_project", None)
        wandb_entity = kwargs.get("wandb_entity", None)
        if wandb_project and wandb_entity:
            self._wandb_run = wandb.init(project=wandb_project, entity=wandb_entity)
        else:
            self._wandb_run = None

    def process_trajectories(self):
        # Step 1: parallel load + policy creation
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(create_empirical_policy, json_file, checkpoint_id): checkpoint_id
                for checkpoint_id, json_file in enumerate(self.json_files)
            }
            results = {}
            for f in as_completed(futures):
                checkpoint_id, trajs, metadata, policy = f.result()
                results[checkpoint_id] = (trajs, metadata, policy)

        # --- First checkpoint (no comparison) ---
        sorted_ids = sorted(results.keys())
        first_id = sorted_ids[0]
        trajectories, metadata, first_policy = results[first_id]

        log_data = {
            "static_graph_metrics": empirical_policy_statistics(first_policy),
            "segmentation_metrics": {
                "segments": 0,
                "unique_segments": 0,
                "clusters": 0,
                "mean_segment_in_cluster": 0.0,
                "mean_unique_segment_in_cluster": 0.0,
                "unique_trajectories": len(set(trajectories))
            }
        }

        if self._wandb_run:
            wandb.config.update(metadata)
            self._wandb_run.log(log_data, step=first_id)

        # Step 2: parallel comparisons (i-1, i)
        tasks = []
        for i in range(1, len(sorted_ids)):
            prev_id = sorted_ids[i - 1]
            curr_id = sorted_ids[i]
            trajectories, metadata, curr_policy = results[curr_id]
            _, _, prev_policy = results[prev_id]
            tasks.append((curr_id, trajectories, metadata, prev_policy, curr_policy))

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_comparison, *task) for task in tasks]
            results_list = [f.result() for f in futures]  # blocking, but done in parallel
            for checkpoint_id, log_data, metadata in sorted(results_list, key=lambda x: x[0]):
                # 🔹 Convert raw PNG bytes -> wandb.Image here
                if "_figs" in log_data:
                    for k, v in log_data["_figs"].items():
                        img = Image.open(io.BytesIO(v))   # convert back to PIL
                        log_data[k] = wandb.Image(img, caption=k)
                    del log_data["_figs"]

                # 🔹 Now safe to log
                if self._wandb_run:
                    wandb.config.update(metadata)
                    self._wandb_run.log(log_data, step=checkpoint_id)


if __name__ == "__main__":
    trajectory_replay = TrajectoryReplay(sys.argv[1],
    wandb_project="agent-trajectory-analysis",
    wandb_entity="ondrej-lukas-czech-technical-university-in-prague"
    )
    trajectory_replay.process_trajectories()
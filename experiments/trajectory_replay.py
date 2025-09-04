from utils.trajectory_utils import load_trajectories_from_json
from utils.trajectory_utils import empirical_policy_statistics
from utils.trajectory_utils import find_trajectory_segments, cluster_segments
from utils.plotting_utils   import plot_segment_cluster_features
from utils.plotting_utils   import plot_trajectory_surprise_matrix, plot_action_per_step_distribution
from utils.plotting_utils   import visualize_clusters, plot_quantile_fan
from utils.trajectory_utils import compute_trajectory_surprises,compute_lambda_returns, policy_comparison
from trajectory import EmpiricalPolicy
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

def load_trajectories(json_file, max_trajectories=None, load_metadata=True):
    """
    Load trajectories from a JSON file.
    Args:
        json_file (str): Path to the JSON file.
        max_trajectories (int, optional): Maximum number of trajectories to load. If None, load all.
        load_metadata (bool): Whether to load metadata from the JSON file.
    Returns:
        list: List of loaded trajectories.
        dict: Metadata dictionary if load_metadata is True, else {}.
    """
    print(f"[load_trajectories] Start loading from {json_file}")
    trajectories, metadata = load_trajectories_from_json(json_file, load_metadata=load_metadata, max_trajectories=max_trajectories)
    print(f"[load_trajectories] Finished loading {len(trajectories)} trajectories from {json_file}")
    return trajectories, metadata

def create_empirical_policy(trajectories, checkpoint_id):
    print(f"[create_empirical_policy] Start creating empirical policy for checkpoint {checkpoint_id}")
    empirical_policy = EmpiricalPolicy(trajectories)
    print(f"[create_empirical_policy] Finished checkpoint {checkpoint_id} ({len(trajectories)} trajectories)")
    return checkpoint_id, empirical_policy

def process_single_trajectory(args):
    """
    Process a single trajectory to compute surprises and segments.
    Args:
        args (tuple): Tuple containing (traj_idx, trajectory, curr_policy, prev_policy, checkpoint_id, per_state_normalization)
    Returns:
        tuple: (segments, surprises)
    """
    traj_idx, t, curr_policy, prev_policy, checkpoint_id, per_state_normalization = args
    rewards = np.array(t.rewards)
    surprises = np.array(compute_trajectory_surprises(t, curr_policy, prev_policy, per_state_normalization, epsilon=1e-12))
    lambda_returns = np.array(compute_lambda_returns(rewards))
    segs = find_trajectory_segments(
        surprises=surprises,
        rewards=rewards,
        lambda_returns=lambda_returns,
        trajectory_id=f"{checkpoint_id}_{traj_idx}"
    )
    return segs, surprises

def process_comparison(checkpoint_id, trajectories, metadata, prev_policy, curr_policy, num_actions=None):
    """
    Process comparison between two empirical policies and segment trajectories.
    Args:
        checkpoint_id (int): Identifier for the current checkpoint.
        trajectories (list): List of trajectories to process.
        metadata (dict): Metadata associated with the trajectories.
        prev_policy (EmpiricalPolicy): Previous empirical policy.
        curr_policy (EmpiricalPolicy): Current empirical policy.
    Returns:
        tuple: (checkpoint_id, log_data, metadata)
    """
    print(f"[process_comparison] Start checkpoint comparison {checkpoint_id}")
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
    
    
    policy_comparison_metrics, js_divergence_per_state = policy_comparison(curr_policy, prev_policy)
    log_data["policy_comparison_metrics"] = policy_comparison_metrics
    segments = []
    surprises = []
    with ThreadPoolExecutor() as pool:
        results = pool.map(
            process_single_trajectory,
            ((traj_idx, t, curr_policy, prev_policy, checkpoint_id, js_divergence_per_state) for traj_idx, t in enumerate(trajectories))
        )
        for segs, traj_surprises in results:
            segments += segs
            surprises.append(traj_surprises)
    # Pad surprises to have shape (num_trajectories, max_len) with np.nan
    max_len = max(len(s) for s in surprises)
    surprises = np.array([np.pad(s, (0, max_len - len(s)), 'constant', constant_values=np.nan) for s in surprises])
    print(f"[process_comparison] Segmentation done for checkpoint {checkpoint_id} ({len(segments)} segments)")

    if segments:
        log_data['segmentation_metrics'].update({
            "segments": len(segments),
            "unique_segments": len({s["features"] for s in segments})
        })

        clustering = cluster_segments(segments)
        print(f"[process_comparison] Clustering done for checkpoint {checkpoint_id} ({len(clustering)} clusters)")
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


        figs = {}

        fig = plot_segment_cluster_features(clustering)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Cluster Feature Summary"] = buf.read()
        plt.close(fig)

        fig = plot_trajectory_surprise_matrix(surprises)
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

        fig, ax = plot_quantile_fan(surprises, num_quantiles=9)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        figs["Quantile Fan Plot"] = buf.read()
        plt.close(fig)

        log_data["_figs"] = figs

    print(f"[process_comparison] Finished checkpoint comparison {checkpoint_id}")
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
        self.params = kwargs
        wandb_project = kwargs.get("wandb_project", None)
        wandb_entity = kwargs.get("wandb_entity", None)
        if wandb_project and wandb_entity:
            self._wandb_run = wandb.init(project=wandb_project, entity=wandb_entity)
        else:
            self._wandb_run = None

    def remap_trajectories(self, trajectories)->dict:
        """
        Re-map custom objects in trajectories to numerical IDs
        """
        remapped_trajectories = trajectories
        return remapped_trajectories
    
    def process_trajectories(self):
        """
        Main processing function to load, segment, cluster and analyze trajectories
        """
        # Load trajectories in parallel using threads
        print("[TrajectoryReplay] Starting parallel loading of checkpoints")
        max_trajectories = self.params.get("max_trajectories", None)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(load_trajectories, json_file, max_trajectories): checkpoint_id
                for checkpoint_id, json_file in enumerate(self.json_files)
            }
            results_trajectories = {}
            for f in as_completed(futures):
                checkpoint_id = futures[f]
                trajs, metadata = f.result()
                print(f"[TrajectoryReplay] Loaded {len(trajs)} trajectories from {checkpoint_id}")
                results_trajectories[checkpoint_id] = (trajs, metadata)
        # Store original trajectories and metadata
        self.original_trajectories = {}
        self.trajectory_metadata = {}
        for cid, (trajs, meta) in results_trajectories.items():
            self.original_trajectories[cid] = trajs
            self.trajectory_metadata[cid] = meta
        # Re-map custom objects to numerical IDs
        self.trajectories = self.remap_trajectories(self.original_trajectories)
        print("[TrajectoryReplay] Finished loading all checkpoints")

        # Create empirical policies in parallel using processes
        print("[TrajectoryReplay] Starting parallel creation of empirical policies")
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(create_empirical_policy, self.original_trajectories[checkpoint_id], checkpoint_id): checkpoint_id
                for checkpoint_id in self.original_trajectories
            }
            results_policies = {}
            for f in as_completed(futures):
                checkpoint_id, policy = f.result()
                print(f"[TrajectoryReplay] Empirical policy created for checkpoint {checkpoint_id}")
                results_policies[checkpoint_id] = policy

        sorted_ids = sorted(results_policies.keys())
        first_id = sorted_ids[0]
        first_policy = results_policies[first_id]
        trajectories = self.trajectories[first_id]

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
            wandb.config.update(self.trajectory_metadata[first_id])
            self._wandb_run.log(log_data, step=first_id)
        num_actions = max(p.num_actions for p in results_policies.values())
        tasks = []
        for i in range(1, len(sorted_ids)):
            prev_id = sorted_ids[i - 1]
            curr_id = sorted_ids[i]
            trajectories = self.trajectories[curr_id]
            metadata = self.trajectory_metadata[curr_id]
            prev_policy = results_policies[prev_id]
            curr_policy = results_policies[curr_id]
            tasks.append((curr_id, trajectories, metadata, prev_policy, curr_policy, num_actions))

        print("[TrajectoryReplay] Starting parallel checkpoint comparisons")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_comparison, *task) for task in tasks]
            results_list = []
            for f in as_completed(futures):
                result = f.result()
                print(f"[TrajectoryReplay] Finished comparison for checkpoint {result[0]}")
                results_list.append(result)
            for checkpoint_id, log_data, metadata in sorted(results_list, key=lambda x: x[0]):
                if "_figs" in log_data:
                    for k, v in log_data["_figs"].items():
                        img = Image.open(io.BytesIO(v))
                        log_data[k] = wandb.Image(img, caption=k)
                    del log_data["_figs"]
                if "test_data" in log_data:
                    surprises = log_data["test_data"].get("surprises", None)
                    rows = [[i, j, surprises[i, j]] for i in range(surprises.shape[0]) for j in range(surprises.shape[1])]
                    log_data["matrix_heatmap"] = wandb.Table(data=rows, columns=["steps", "trajectories", "surprise"])
                    del log_data["test_data"]

                if self._wandb_run:
                    wandb.config.update(metadata)
                    self._wandb_run.log(log_data, step=checkpoint_id)


if __name__ == "__main__":
    trajectory_replay = TrajectoryReplay(sys.argv[1],
    wandb_project="agent-trajectory-analysis",
    wandb_entity="ondrej-lukas-czech-technical-university-in-prague"
    )
    trajectory_replay.process_trajectories()
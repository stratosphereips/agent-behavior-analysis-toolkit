from utils.trajectory_utils import get_trajectory_action_ngrams, load_trajectories
from utils.trajectory_utils import empirical_policy_statistics
from utils.trajectory_utils import find_trajectory_segments, cluster_segments
from utils.plotting_utils   import plot_segment_cluster_features, plot_action_per_step_distribution
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
    # bigrams = get_trajectory_action_ngrams(t, n=2)
    surprises = np.array(compute_trajectory_surprises(t, curr_policy, prev_policy, per_state_normalization, epsilon=1e-12))
    lambda_returns = np.array(compute_lambda_returns(rewards))
    actions = np.array(t.actions)
    states = np.array(t.states)
    segs = find_trajectory_segments(
        surprises=surprises,
        rewards=rewards,
        lambda_returns=lambda_returns,
        actions=actions,
        states=states,
        trajectory_id=f"{checkpoint_id}_{traj_idx}"
    )
    return segs, surprises

def process_comparison(checkpoint_id, trajectories, metadata, prev_policy, curr_policy, all_actions:set, num_actions:int=None):
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
    
    
    # policy_comparison_metrics, js_divergence_per_state = policy_comparison(curr_policy, prev_policy, all_actions)
    # log_data["policy_comparison_metrics"] = policy_comparison_metrics
    # segments = []
    # surprises = []
    # ngrams = []
    # with ThreadPoolExecutor() as pool:
    #     results = pool.map(
    #         process_single_trajectory,
    #         ((traj_idx, t, curr_policy, prev_policy, checkpoint_id, js_divergence_per_state) for traj_idx, t in enumerate(trajectories))
    #     )
    #     for segs, traj_surprises, in results:
    #         segments += segs
    #         surprises.append(traj_surprises)

    # max_len = max(len(s) for s in surprises)
    # # pre-allocate matrix and fill with np.nan
    # surprise_matrix = np.full((len(surprises), max_len), np.nan)
    # # insert each surprise array into the matrix
    # for i, s in enumerate(surprises):
    #     surprise_matrix[i, :len(s)] = s
    # print(f"[process_comparison] Segmentation done for checkpoint {checkpoint_id} ({len(segments)} segments)")
    figs = {}
    # action_to_id = {a: i for i, a in enumerate(curr_policy.actions)}
    num_actions = len(all_actions) if num_actions is None else num_actions

    # Action distribution plot
    fig = plot_action_per_step_distribution(trajectories, num_actions, normalize=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    figs["Action Distribution Plot"] = buf.read()
    plt.close(fig)

    # # surprise heatmap
    # fig = plot_trajectory_surprise_matrix(surprise_matrix)
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    # buf.seek(0)
    # figs["Segment Surprise Plot"] = buf.read()
    # plt.close(fig)
    
    # # quantile fan plot
    # fig, ax = plot_quantile_fan(surprise_matrix, num_quantiles=9)
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    # buf.seek(0)
    # figs["Quantile Fan Plot"] = buf.read()
    # plt.close(fig)

    # -----------------------------
    # NEW: Normalized Surprise Plot
    # -----------------------------
    # We want to plot the mean and std dev of the "normalized surprise" across all trajectories in this checkpoint.
    # We first need to compute it.
    
    # 1. Compute per-state JS-divergence (normalization factor)
    js_div_per_state, mean_js = js_divergence_per_state(curr_policy, prev_policy, all_actions)
    
    # 2. Compute surprises for all trajectories
    all_surprises = []
    for t in trajectories:
        # compute_trajectory_surprises returns a list of surprises for each step
        surprises = compute_trajectory_surprises(t, curr_policy, prev_policy, js_div_per_state)
        all_surprises.extend(surprises)
    
    if all_surprises:
        mean_surprise = np.mean(all_surprises)
        std_surprise = np.std(all_surprises)
        log_data["mean_action_surprise"] = mean_surprise
        log_data["std_action_surprise"] = std_surprise
        print(f"[process_comparison] Checkpoint {checkpoint_id}: Mean Surprise = {mean_surprise:.4f}, Std = {std_surprise:.4f}")
    else:
        log_data["mean_action_surprise"] = 0.0
        log_data["std_action_surprise"] = 0.0

    # # if len(ngrams) > 0:
    # #     ngram_matrix = np.zeros((num_actions, num_actions), dtype=int)
    # #     for ngram in ngrams:
    # #         for a1, a2 in ngram:
    # #             ngram_matrix[action_to_id[a1], action_to_id[a2]] += 1
    # #     print(f"[process_comparison] Action bigram matrix computed for checkpoint {checkpoint_id} done.")
    # #     fig = plot_sankey_plotly(ngram_matrix, [str(a) for a in action_to_id.keys()])
    # #     buf = io.BytesIO()
    # #     fig.write_image(buf, format="png")
    # #     buf.seek(0)
    # #     figs["Action Bigram Chord Plot"] = buf.read()
    
    
    # if segments:
    #     log_data['segmentation_metrics'].update({
    #         "segments": len(segments),
    #         "unique_segments": len({s["features"] for s in segments})
    #     })

    #     clustering = cluster_segments(segments)
    #     print(f"[process_comparison] Clustering done for checkpoint {checkpoint_id} ({len(clustering)} clusters)")
    #     segments_per_cluster = [len(segs) for segs in clustering.values()]
    #     unique_segments_per_cluster = [
    #         len({seg["features"] for seg in segs})
    #         for segs in clustering.values()
    #     ]

    #     log_data['segmentation_metrics'].update({
    #         "clusters": len(clustering),
    #         "mean_segment_in_cluster": np.mean(segments_per_cluster) if segments_per_cluster else 0.0,
    #         "mean_unique_segment_in_cluster": np.mean(unique_segments_per_cluster) if unique_segments_per_cluster else 0.0
    #     })


    #     fig = plot_segment_cluster_features(clustering)
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png")
    #     buf.seek(0)
    #     figs["Cluster Feature Summary"] = buf.read()
    #     plt.close(fig)

    #     unique_trajectories = set(trajectories)
    #     max_len = max(len(t) for t in unique_trajectories)
    #     fig = visualize_clusters(clustering, max_len)
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png", bbox_inches="tight")
    #     buf.seek(0)
    #     figs["Cluster Visualization"] = buf.read()
    #     plt.close(fig)

    #     fig, ax = plot_quantile_fan(surprise_matrix, num_quantiles=9)
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png")
    #     buf.seek(0)
    #     figs["Quantile Fan Plot"] = buf.read()
    #     plt.close(fig)

    #     fig = plot_cluster_distribution_per_step(clustering, max_len,)
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png")
    #     buf.seek(0)
    #     figs["Cluster Distribution Plot"] = buf.read()

    #     plt.close(fig)
            # fig = plot_combined_trajectory_analysis(
            #     trajectories,
            #     num_actions,
            #     clusters=clustering,
            #     trajectory_len=max_len,
            #     surprise_matrix=surprise_matrix
            # )
            # buf = io.BytesIO()
            # fig.savefig(buf, format="png", bbox_inches="tight")
            # buf.seek(0)
            # figs["Combined Trajectory Analysis"] = buf.read()
            # plt.close(fig)
    log_data["_figs"] = figs

    print(f"[process_comparison] Finished checkpoint comparison {checkpoint_id}")
    return checkpoint_id, log_data, metadata

class TrajectoryReplay:
    """
    Class for loading and processing of recorded trajectories
    """

    def __init__(self, trajectory_dir, env="numpy", **kwargs):
        self.trajectory_dir = trajectory_dir
        self.env = env
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
        all_actions = set()
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(load_trajectories, json_file, max_trajectories, True, self.env): checkpoint_id
                for checkpoint_id, json_file in enumerate(self.json_files)
            }
            results_trajectories = {}
            for f in as_completed(futures):
                checkpoint_id = futures[f]
                trajs, metadata = f.result()
                print(f"[TrajectoryReplay] Loaded {len(trajs)} trajectories from {checkpoint_id}")
                results_trajectories[checkpoint_id] = (trajs, metadata)
                for traj in trajs:
                    all_actions.update(traj.actions)
        print(f"[TrajectoryReplay] Found {len(all_actions)} unique actions across all checkpoints")
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
            tasks.append((curr_id, trajectories, metadata, prev_policy, curr_policy, all_actions,num_actions))

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

        # -----------------------------
        # NEW: Plot Mean Action Surprise across Checkpoints
        # -----------------------------
        surprises_mean = []
        surprises_std = []
        checkpoint_ids = []

        # Sort again to ensure order
        sorted_results = sorted(results_list, key=lambda x: x[0])
        
        for checkpoint_id, log_data, _ in sorted_results:
             if "mean_action_surprise" in log_data:
                 surprises_mean.append(log_data["mean_action_surprise"])
                 surprises_std.append(log_data["std_action_surprise"])
                 
                 # Try to extract integer from checkpoint string for better plotting
                 try:
                     # Assumes format like "cp_12345" or just "12345"
                     if "_" in str(checkpoint_id):
                        cid_int = int(str(checkpoint_id).split("_")[-1])
                     else:
                        cid_int = int(str(checkpoint_id))
                     checkpoint_ids.append(cid_int)
                 except ValueError:
                     checkpoint_ids.append(checkpoint_id)

        if checkpoint_ids:
            plt.figure(figsize=(12, 6))
            
            # Sort by checkpoint ID integer if possible
            if all(isinstance(c, int) for c in checkpoint_ids):
                sorted_indices = np.argsort(checkpoint_ids)
                x_vals = np.array(checkpoint_ids)[sorted_indices]
                y_mean = np.array(surprises_mean)[sorted_indices]
                y_std = np.array(surprises_std)[sorted_indices]
            else:
                x_vals = checkpoint_ids
                y_mean = surprises_mean
                y_std = surprises_std

            plt.plot(x_vals, y_mean, label='Mean Normalized Surprise', marker='o', color='blue')
            plt.fill_between(x_vals, 
                             np.array(y_mean) - np.array(y_std), 
                             np.array(y_mean) + np.array(y_std), 
                             color='blue', alpha=0.2, label='Std Dev')
            
            plt.xlabel('Checkpoint')
            plt.ylabel('Normalized Surprise (log-ratio / JS)')
            plt.title('Mean Action Surprise across Checkpoints')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plot_filename = 'mean_action_surprise_summary.png'
            plot_path = os.path.join(self.trajectory_dir, plot_filename)
            plt.savefig(plot_path)
            print(f"[TrajectoryReplay] Saved summary plot to {plot_path}")
            
            if self._wandb_run:
                self._wandb_run.log({"mean_action_surprise_summary": wandb.Image(plot_path)})


if __name__ == "__main__":
    trajectory_replay = TrajectoryReplay(sys.argv[1],
    wandb_project="agent-trajectory-analysis",
    wandb_entity="ondrej-lukas-czech-technical-university-in-prague",
    max_trajectories=1000
    )
    trajectory_replay.process_trajectories()
from utils.trajectory_utils import load_trajectories_from_json
from utils.trajectory_utils import empirical_policy_statistics
from utils.trajectory_utils import find_trajectory_segments, cluster_segments
from utils.plotting_utils   import plot_segment_cluster_features, plot_trajectory_network_colored_nodes_by_cluster
from utils.plotting_utils   import plot_trajectory_segments, plot_action_per_step_distribution, plot_trajectory_heatmap
from utils.plotting_utils   import plot_trajectory_heatmap
from utils.trajectory_utils import get_trajectory_action_change
from utils.trajectory_utils import get_clusters_per_step
from utils.trajectory_utils import compute_trajectory_surprises
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


class TrajectoryReplay:
    """
    Class for loading and processing of recorded trajectories
    """

    def __init__(self, trajectory_dir, **kwargs):
        self.trajectories = []
        self.json_files = sorted([os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir) if f.endswith(".json")])
        print(f"Found {len(self.json_files)} JSON files in {trajectory_dir}")
        self._previous_policy = None
        wandb_project = kwargs.get("wandb_project", None)
        wandb_entity = kwargs.get("wandb_entity", None)
        if wandb_project and wandb_entity:
            self._wandb_run = wandb.init(project=wandb_project, entity=wandb_entity)
        else:
            self._wandb_run = None

    def process_trajectories(self):
        for checkpoint_id, json_file in enumerate(self.json_files):
            trajectories, trajectory_metadata = load_trajectories_from_json(json_file, load_metadata=True, max_trajectories=200)
            print(f"Loaded {len(trajectories)} trajectories from {json_file}: - checkpoint {checkpoint_id}")
            self.trajectories.extend(trajectories)
            empirical_policy = EmpiricalPolicy(trajectories)
            print(f"Total_actions: {empirical_policy.num_actions}")
            print(trajectory_metadata)


            log_data = {"static_graph_metrics":empirical_policy_statistics(empirical_policy)}
            log_data["Cluster Feature Summary"] = None
            log_data["Segment Surprise Plot"] = None
            log_data['segmentation_metrics'] = {     # Initialize with default/zero values
                "segments": 0,
                "unique_segments": 0,
                "clusters": 0,
                "mean_segment_in_cluster": 0.0,
                "mean_unique_segment_in_cluster": 0.0,
                "unique_trajectories":len(set(trajectories))
            }
            if self._previous_policy:
                segments = []
                for t in trajectories:
                    new_segments = find_trajectory_segments(trajectory=t, policy=empirical_policy, previous_policy=self._previous_policy)
                    segments += new_segments
                if len(segments) > 0:
                    log_data['segmentation_metrics']["segments"] = len(segments)
                    log_data['segmentation_metrics']["unique_segments"] = len(set([tuple(s["features"].values()) for s in segments]))
                    clustering  = cluster_segments(segments)
                    segments_per_cluster = []
                    unique_segments_per_cluster = []
                    unique_segment_clusters = {}
                    for cluster_id, segments in clustering.items():

                        segments_per_cluster.append(len(segments))
                        s = set()
                        unique_segment_clusters[cluster_id] = []
                        for segment in segments:
                            if tuple(segment["features"].values()) not in s:
                                unique_segment_clusters[cluster_id].append(segment)
                                s.add(tuple(segment["features"].values()))
                        unique_segments_per_cluster.append(len(s))
                    if self._wandb_run:
                        cluster_summary_fig = plot_segment_cluster_features(clustering)
                        log_data["Cluster Feature Summary"] = wandb.Image(cluster_summary_fig, caption="Feature Summary per cluster")
                        plt.close(cluster_summary_fig)  # clean up
                        segments_plot = plot_trajectory_segments(trajectories, empirical_policy, self._previous_policy,f"Surprise_plot_cp_{checkpoint_id}")
                        log_data["Segment Surprise Plot"]  = wandb.Image(segments_plot, caption="Surprise per step in all trajectories")
                        plt.close(segments_plot) #cleanup
                        log_data['segmentation_metrics']["clusters"] = len(clustering)
                        log_data['segmentation_metrics']["mean_segment_in_cluster"] = np.mean(segments_per_cluster)
                        log_data['segmentation_metrics']["mean_unique_segment_in_cluster"] = np.mean(unique_segments_per_cluster)
                        action_distribution_plot = plot_action_per_step_distribution(trajectories, 2, normalize=True)
                        log_data["Action Distribution Plot"] = wandb.Image(action_distribution_plot, caption="Action Distribution per Time Step")
                        plt.close(action_distribution_plot)  # cleanup
                        
                        unique_trajectories = set(trajectories)
                        max_len = max(len(t) for t in unique_trajectories)
                        surprises = np.zeros((len(unique_trajectories), max_len))
                        action_change = np.zeros((len(unique_trajectories), max_len))
                        cluster_per_step = np.zeros((len(unique_trajectories), max_len))
                        for i, trajectory in enumerate(unique_trajectories):
                            surprises[i, :len(trajectory)] = compute_trajectory_surprises(trajectory, empirical_policy, self._previous_policy)
                            action_change[i, :len(trajectory)] = get_trajectory_action_change(trajectory, empirical_policy, self._previous_policy)
                            cluster_per_step[i, :len(trajectory)] = get_clusters_per_step(trajectory, clustering)
                        trajectory_heatmap_plot = plot_trajectory_heatmap(surprises, action_change, cluster_per_step)[0]
                        # trajectory_plot = plot_trajectory_network_colored_nodes_by_cluster(trajectories[0], unique_segment_clusters)

                        log_data["Trajectory Combined Heatmap"] = wandb.Image(trajectory_heatmap_plot, caption="Trajectory Combined Heatmap")
                        plt.close(trajectory_heatmap_plot)
                        #trajectory_plot = plot_trajectory_network_colored_nodes_by_cluster(trajectories[0], unique_segment_clusters)

                        #log_data["Trajectory Network Plot"] = wandb.Image(Image.open(io.BytesIO(trajectory_plot)), caption="Trajectory Network Colored by Cluster") # cleanup
                    print(log_data)
            if self._wandb_run:
                wandb.config.update(trajectory_metadata)
                self._wandb_run.log(log_data,step=checkpoint_id)
            self._previous_policy = empirical_policy

if __name__ == "__main__":
    trajectory_replay = TrajectoryReplay(sys.argv[1],
    wandb_project="agent-trajectory-analysis",
    wandb_entity="ondrej-lukas-czech-technical-university-in-prague")
    trajectory_replay.process_trajectories()
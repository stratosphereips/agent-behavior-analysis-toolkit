import argparse
import numpy as np
import wandb
from typing import Iterable
from trajectory import EmpiricalPolicy
import matplotlib.pyplot as plt
from utils.trajectory_utils import empirical_policy_statistics, find_trajectory_segments, cluster_segments
from utils.plotting_utils import plot_trajectory_segments, plot_segment_cluster_features, plot_action_per_step_distribution
class ReplayBuffer:

    def __init__(self, env_name:str, agent_name:str, **kwargs):
        self.params = kwargs
        self.experiment_info = {
            "env_name": env_name,
            "agent_name": agent_name,
        
        }
        self.wandb_run = kwargs.get("wandb_run", None)
        self.checkpoints = []
        self._previous_policy = None



    def add_trajectories(self, trajectories: Iterable) -> None:
        # start dict for log data
        log_data = {}
        # store the trajectories
        self.checkpoints.append(trajectories)
        # build empirical policy
        empirical_policy = EmpiricalPolicy(trajectories)
        log_data = {"static_graph_metrics":empirical_policy_statistics(empirical_policy, is_win_fn=self.params.get("is_win_fn", None))}
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
        if self._previous_policy: # there is something to comapre with (previous checkpoint)
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
                        for cluster_id, segments in clustering.items():
                            segments_per_cluster.append(len(segments))
                            s = set()
                            for segment in segments:
                                s.add(tuple(segment["features"].values()))
                            unique_segments_per_cluster.append(len(s))
                        cluster_summary_fig = plot_segment_cluster_features(clustering)
                        log_data["Cluster Feature Summary"] = wandb.Image(cluster_summary_fig, caption="Feature Summary per cluster")
                        plt.close(cluster_summary_fig)  # clean up
                        segments_plot = plot_trajectory_segments(trajectories, empirical_policy, self._previous_policy,f"Surprise_plot_cp_{self._chechpoint_id}")
                        log_data["Segment Surprise Plot"]  = wandb.Image(segments_plot, caption="Surprise per step in all trajectories")
                        plt.close(segments_plot) #cleanup
                        log_data['segmentation_metrics']["clusters"] = len(clustering)
                        log_data['segmentation_metrics']["mean_segment_in_cluster"] = np.mean(segments_per_cluster)
                        log_data['segmentation_metrics']["mean_unique_segment_in_cluster"] = np.mean(unique_segments_per_cluster)
                        action_distribution_plot = plot_action_per_step_distribution(trajectories, self.params.get("num_actions", 2), normalize=True)
                        log_data["Action Distribution Plot"] = wandb.Image(action_distribution_plot, caption="Action Distribution per Time Step")
                        plt.close(action_distribution_plot)  # cleanup
                        print(log_data)
        if self.wandb_run is not None:
            self.wandb_run.log(log_data,step=len(self.checkpoints)-1)
        self._previous_policy = empirical_policy
    
    if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("a", default=4242, type=int, help="Random seed.")
        args = parser.parse_args()
    
        ReplayBuffer = ReplayBuffer(
            env_name="AiDojo",
            agent_name="Random",

        )
        # Fix random seed
        experiment_config = {
            "env": "MountainCar-v0-discrete",
            "model": "Q-learning",
        }
        experiment_config.update(vars(args))
        print(experiment_config)
        wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="ondrej-lukas-czech-technical-university-in-prague",
            # Set the wandb project where this run will be logged.
            project="agent-trajectory-analysis",
            # Track hyperparameters and run metadata.
            config=experiment_config,
        ) 
import numpy as np
from trajectory_graph import TrajectoryGraph, Transition, plot_tg_mdp
from trajectory import EmpiricalPolicy
from utils.trajectory_utils import get_motifs, empirical_policy_statistics, find_trajectory_segments, cluster_segments
from utils.plotting_utils import plot_trajectory_segments, plot_segment_cluster_features, plot_action_per_step_distribution
import wandb
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, **kwargs):
        self.params = kwargs  # Store parameters in a dictionary       
        self._initialize_agent()
        self.wandb_run = kwargs.get("wandb_run", None)
        self._chechpoint_id = 0
        self._previous_policy = None
        self._motifs = set()
        if self.wandb_run:
            self.tg = None

    def _initialize_agent(self):
        """
        Optional: Allows subclasses to do specific initializations using self.params
        """
        pass

    def step(state, training=False):
        raise NotImplementedError
    
    def evaluate_policy(self, env, num_episodes:int, get_trajectories:bool=False, final_evaluation=False):
        returns = []
        trajectories = None
        # enable trajectory tracking
        env.start_recording()
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            ret = 0
            while not done:
                action = self.step(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward
                state = next_state
            returns.append(ret)
        if self.wandb_run is not None:
            if not final_evaluation:
                trajectories = env.trajectory_log              
                empirical_policy = EmpiricalPolicy(trajectories)
                log_data = {"static_graph_metrics":empirical_policy_statistics(empirical_policy, is_win_fn=env.is_win_fn)}
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
                        action_distribution_plot = plot_action_per_step_distribution(trajectories, env.action_space.n, normalize=True)
                        log_data["Action Distribution Plot"] = wandb.Image(action_distribution_plot, caption="Action Distribution per Time Step")
                        plt.close(action_distribution_plot)  # cleanup
                        print(log_data)
                self.wandb_run.log(log_data,step=self._chechpoint_id)
                self._previous_policy = empirical_policy
                self._chechpoint_id += 1
        # stop the trajectory recording
        env.stop_recording()
        env.clear_trajectory_log()
        return returns, trajectories

    def train_policy(self, env, num_episodes , evaluate_each=None, evaluate_for=None):
        raise NotImplementedError
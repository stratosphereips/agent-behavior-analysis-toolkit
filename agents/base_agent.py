from utils.trajectory_utils import store_trajectories_to_json, numpy_default
import os
import numpy as np


class Agent:
    def __init__(self, trajectory_json_default_fn=None,**kwargs):
        self.params = kwargs  # Store parameters in a dictionary)
        self._initialize_agent()
        self.trajectory_json_encoder = trajectory_json_default_fn if trajectory_json_default_fn else numpy_default# Use NumpyEncoder as default
        self.wandb_run = kwargs.get("wandb_run", None)
        self.store_trajectories = kwargs.get("store_trajectories", False)
        self._checkpoint_id = 0

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
                action = self.step(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward
                state = next_state
            returns.append(ret)
        if self.store_trajectories:
            if not final_evaluation:
                trajectories = env.trajectory_log
                metadata = {k:v for k,v in self.params.get("experiment_config", {}).items()}
                metadata["action_space_size"] = env.action_space.n
                metadata["observation_space_size"] = env.observation_space.n

                # Create folder based on env and model
                foldername = ""
                try:
                    foldername += metadata["experiment_config"]["env"]
                except KeyError:
                    pass
                try:
                    foldername = foldername + f"_{metadata['experiment_config']['model']}"
                except KeyError:
                    pass

                foldername = f"trajectories/{foldername}"
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                # Create nested_folder based on parameters
                try:
                    nested_foldername = "_".join(f"{k}={metadata["experiment_config"][k]}" for k in sorted(metadata["experiment_config"].keys()) if k not in ["env", "model"])
                except KeyError:
                    nested_foldername = ""
                
                foldername = f"{foldername}/{nested_foldername}"
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                filename = f"cp_{self._checkpoint_id:02d}.json"
                store_trajectories_to_json(trajectories, f"{foldername}/{filename}", metadata=metadata, encoder=self.trajectory_json_encoder)
        if self.wandb_run is not None:
            self.wandb_run.log({"mean_return": np.mean(returns) if returns else 0}, step=self._checkpoint_id)
        self._checkpoint_id += 1
        # stop the trajectory recording
        env.stop_recording()
        env.clear_trajectory_log()
        return returns, trajectories

    def train_policy(self, env, num_episodes , evaluate_each=None, evaluate_for=None):
        raise NotImplementedError
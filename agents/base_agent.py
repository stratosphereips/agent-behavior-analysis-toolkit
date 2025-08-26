from utils.trajectory_utils import store_trajectories_to_json
from utils.trajectory_utils import NumpyEncoder
import os


class Agent:
    def __init__(self, **kwargs):
        self.params = kwargs  # Store parameters in a dictionary)
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
                action = self.step(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ret += reward
                state = next_state
            returns.append(ret)
        if self.wandb_run is not None:
            if not final_evaluation:
                trajectories = env.trajectory_log
                metadata = {k:v for k,v in self.params.items() if k not in ["wandb_run", "discretized_env.observation_space.n", "discretized_env.action_space.n"]}
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
                filename = f"cp_{self._chechpoint_id:02d}.json"
                store_trajectories_to_json(trajectories, f"{foldername}/{filename}", metadata=metadata, encoder=NumpyEncoder)
                self._chechpoint_id += 1
        # stop the trajectory recording
        env.stop_recording()
        print(env.starting_states)
        env.clear_trajectory_log()
        return returns, trajectories

    def train_policy(self, env, num_episodes , evaluate_each=None, evaluate_for=None):
        raise NotImplementedError
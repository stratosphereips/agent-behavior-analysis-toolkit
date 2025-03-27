import numpy as np
from trajectory_graph import TrajectoryGraph, Transition

class Agent:
    def __init__(self, **kwargs):
        self.params = kwargs  # Store parameters in a dictionary       
        self._initialize_agent()
        self.wandb_run = kwargs.get("wandb_run", None)
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
        trajectories = []
        returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            ret = 0
            t = []
            while not done:
                action = self.step(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                t.append(Transition(state, action, reward, next_state))
                done = terminated or truncated
                ret += reward
                state = next_state
            trajectories.append(t)
            returns.append(ret)
        if self.wandb_run is not None:
            if not final_evaluation:
                tg = TrajectoryGraph()
                for t in trajectories:
                    tg.add_trajectory(t)
                log_data = {"static_graph_metrics":tg.get_graph_metrics()}
                if self.tg:
                    log_data["tg_diff"] = tg.compare_with_previous(self.tg)
                self.wandb_run.log(log_data)
                self.tg = tg
        return returns, trajectories

    def train_policy(self, env, num_episodes , evaluate_each=None, evaluate_for=None):
        raise NotImplementedError
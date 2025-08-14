import numpy as np
from .base_agent import Agent
import math

class RandomAgent(Agent):
    
    def __init__(self, acion_space_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space_size = acion_space_size

    def step(self, state, training=False):
        return np.random.choice(self.action_space_size)

    def train_policy(self, env, num_episodes:int, evaluate_each:int, evaluate_for:int):
        """Q_learning algorithm with periodic evaluation."""
        eval_results = []  # Store evaluation results
        for episode in range(1, num_episodes + 1):
            # Periodic evaluation
            if evaluate_each:
                if episode % evaluate_each == 0:
                    returns,_ = self.evaluate_policy(env, evaluate_for)
                    mean_ret =  np.mean(returns)
                    eval_results.append((episode, mean_ret))
                    print(f"Episode {episode}: Mean return = {mean_ret:.2f}")
        return eval_results
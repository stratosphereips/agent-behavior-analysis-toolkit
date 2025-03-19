import numpy as np
import gymnasium as gym
from utils import get_bins, discretize_observation


class DiscretizedCartPoleEnv(gym.ObservationWrapper):
    """
    A wrapper that discretizes the observation space of CartPole.
    """

    def __init__(self, num_bins=20, velocity_bound=5.0, angular_velocity_bound=5.0):
        env = gym.make("CartPole-v1")
        super().__init__(env)

        self.num_bins = num_bins
        self.velocity_bound = velocity_bound
        self.angular_velocity_bound = angular_velocity_bound

        # Define the boundaries of the observations.
        observation_space_high = env.observation_space.high
        observation_space_low = env.observation_space.low

        observation_space_high[1] = velocity_bound
        observation_space_high[3] = angular_velocity_bound
        observation_space_low[1] = -velocity_bound
        observation_space_low[3] = -angular_velocity_bound
        self.bins = get_bins(observation_space_high, observation_space_low, num_bins=20)

    def observation(self, observation):
        """Discretizes the observation."""
        return discretize_observation(observation, self.bins)


if __name__ == "__main__":
    discretized_env = DiscretizedCartPoleEnv(num_bins=50)

    observation, info = discretized_env.reset()
    print(f"Discretized observation: {observation}")

    # Now you can use discretized_env with your Q-learning or other tabular method.
    # For example:
    action = discretized_env.action_space.sample()
    next_observation, reward, terminated, truncated, info = discretized_env.step(action)
    print(f"Next Discretized Observation: {next_observation}")
    discretized_env.close()
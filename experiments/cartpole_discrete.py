from environments.gym_discerete_wrappers import DiscreteCartPoleWrapper
from environments.gym_discretization_wrapper import EvaluationEnv, CustomEnv
from agents.q_learning import Qlearning
import gymnasium as gym
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    args = parser.parse_args()
    # Fix random seed
    np.random.seed(args.seed)
    discretized_env = CustomEnv(DiscreteCartPoleWrapper(gym.make("CartPole-v1"), bins=16), seed=args.seed)
    agent = Qlearning(
        discretized_env.observation_space.n,
        discretized_env.action_space.n,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.8,
        epsilon_min=0.01,
        epsilon_decay=0.995
        )
    agent.train_policy(discretized_env, num_episodes=50000, evaluate_each=2500, evaluate_for=1000)
    # Final evaluation
    agent.evaluate_policy(discretized_env, num_episodes=2000)
    
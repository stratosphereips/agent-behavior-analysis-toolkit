
from environments.gym_discretization_wrapper import CustomEnv
from agents.dqn import DQN
from agents.reinforce import REINFORCE
import gymnasium as gym
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    args = parser.parse_args()
    # Fix random seed
    np.random.seed(args.seed)
    env = gym.make("CartPole-v1")
    # env = ClipObservation(env)
    # env = RescaleObservation(env, min_obs=-1, max_obs=1)
    env = CustomEnv(env, seed=args.seed)

    #agent = DQN(env.observation_space.shape[0], env.action_space.n)
    agent = REINFORCE(env.observation_space.shape[0], env.action_space.n)
    agent.train_policy(env, num_episodes=1000, evaluate_each=100, evaluate_for=200)
    # Final evaluation
    returns, _ = agent.evaluate_policy(env, num_episodes=2000)
    print(f"Final evaluation:{np.mean(returns):.2f}+-{np.std(returns):.2f}")
    
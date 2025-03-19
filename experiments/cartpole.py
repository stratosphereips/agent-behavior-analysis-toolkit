
from environments.gym_discretization_wrapper import CustomEnv
from agents.dqn import DQN
import gymnasium as gym
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    args = parser.parse_args()
    # Fix random seed
    np.random.seed(args.seed)
    env = CustomEnv(gym.make("CartPole-v1"), seed=args.seed)

    agent = DQN(env.observation_space.shape[0], env.action_space.n)
    agent.train_policy(env, num_episodes=5000, evaluate_each=500, evaluate_for=500)
    # Final evaluation
    returns, _ = agent.evaluate_policy(env, num_episodes=2000)
    print(f"Final evaluation:{np.mean(returns):.2f}+-{np.std(returns):.2f}")
    
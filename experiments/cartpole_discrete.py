from environments.gym_discerete_wrappers import DiscreteCartPoleWrapper
from environments.gym_discretization_wrapper import EvaluationEnv, CustomEnv
from agents.q_learning import Qlearning
from agents.sarsa import Sarsa
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
    #discretized_env = CustomEnv(DiscreteCartPoleWrapper(gym.make("CartPole-v1"), bins=8), seed=args.seed)
    discretized_env = CustomEnv(gym.make("CartPole-v1"), seed=args.seed)
    
    # agent = Qlearning(
    #     discretized_env.observation_space.n,
    #     discretized_env.action_space.n,
    #     alpha=0.1,
    #     gamma=0.95,
    #     epsilon=1,
    #     epsilon_min=0.01,
    #     epsilon_decay=0.0001
    #     )

    # agent = Sarsa(
    #     discretized_env.observation_space.n,
    #     discretized_env.action_space.n,
    #     alpha=0.1,
    #     gamma=0.95,
    #     epsilon=1,
    #     epsilon_min=0.05,
    #     epsilon_decay=0.0005
    # )
    print(discretized_env.action_space.shape)
    agent = DQN(discretized_env.observation_space.shape[0], discretized_env.action_space.n)
    


    agent.train_policy(discretized_env, num_episodes=5000, evaluate_each=500, evaluate_for=500)
    # Final evaluation
    returns, _ = agent.evaluate_policy(discretized_env, num_episodes=2000)
    print(f"Final evaluation:{np.mean(returns):.2f}+-{np.std(returns):.2f}")
    
from environments.gym_discerete_wrappers import DiscreteCartPoleWrapper
from agents.random import RandomAgent
from agents.ppo import PPOAgent
from agents.dqn import DQNAgent
from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent

import gymnasium as gym
import argparse
import numpy as np
import wandb
import random
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=4242, type=int, help="Random seed.")
    parser.add_argument("--episodes", default=1000, type=int, help="Number of training episodes")
    parser.add_argument("--evaluate_each", default=500, type=int, help="Periodic evluation frequency")
    parser.add_argument("--evaluate_for", default=500, type=int, help="Periodic evluation length")
    parser.add_argument("--model", default="random", type=str, choices=["random", "ppo", "dqn", "sarsa", "q_learning"], help="Agent model type")
    
    # Hyperparameters (Optional overrides)
    parser.add_argument("--alpha", type=float, help="Learning rate (tabular)")
    parser.add_argument("--lr", type=float, help="Learning rate (NN)")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Epsilon (exploration)")
    parser.add_argument("--epsilon_min", type=float, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, help="Epsilon decay rate/factor")
    
    args = parser.parse_args()

    custom_win_fn = lambda trajectory: len(trajectory) > 450
    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # Agent-specific configurations
    AGENT_CONFIGS = {
        "ppo": {
            "lr": 3e-4,
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "train_iters": 10,
            "batch_size": 64,
            "value_coef": 0.5,
            "entropy_coef": 0.01
        },
        "dqn": {
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 64,
            "memory_size": 10000,
            "replay_each": 4,
            "target_update_every": 1000
        },
        "sarsa": {
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9995 
        },
        "q_learning": {
             "alpha": 0.1,
             "gamma": 0.99,
             "epsilon": 1.0,
             "epsilon_min": 0.01,
             "epsilon_decay": 0.9995 # Tabular often needs slower decay
        },

        "random": {}
    }

    # Start with the specific config for the selected model
    experiment_config = AGENT_CONFIGS[args.model].copy()
    
    # Add common/environment configs
    experiment_config.update({
        "env": "CartPole-v1-discrete",
        "model": args.model,
    })
    
    # Override with any CLI arguments that were explicitly set (not None)
    experiment_config.update({k: v for k, v in vars(args).items() if v is not None})
    print(experiment_config)
    
    # basic env
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    # Add Discretization Layer
    discretized_env = DiscreteCartPoleWrapper(env, bins=[1, 4, 6, 4])
    
    if args.model == "random":
        agent = RandomAgent(env=discretized_env, store_trajectories=True, args=experiment_config)
    elif args.model == "ppo":
        agent = PPOAgent(env=discretized_env, store_trajectories=True, args=experiment_config)
    elif args.model == "dqn":
        agent = DQNAgent(env=discretized_env, store_trajectories=True, args=experiment_config)
    elif args.model == "sarsa":
        agent = SarsaAgent(env=discretized_env, store_trajectories=True, args=experiment_config)
    elif args.model == "q_learning":
        agent = QLearningAgent(env=discretized_env, store_trajectories=True, args=experiment_config)

        
    eval_results = agent.train_policy(discretized_env, num_episodes=experiment_config["episodes"], evaluate_each=experiment_config["evaluate_each"], evaluate_for=experiment_config["evaluate_for"])
    print(eval_results)
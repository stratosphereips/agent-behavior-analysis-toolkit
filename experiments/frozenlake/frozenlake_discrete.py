import gymnasium as gym
import argparse
import numpy as np
import os
import random

# Configure Tensorflow to use memory growth to allow multiple concurrent runs
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

from agents.random import RandomAgent
from agents.ppo import PPOAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.dqn import DQNAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=4242, type=int, help="Random seed.")
    parser.add_argument("--episodes", default=30000, type=int, help="Number of training episodes")
    parser.add_argument("--evaluate_each", default=500, type=int, help="Periodic evluation frequency")
    parser.add_argument("--evaluate_for", default=500, type=int, help="Periodic evluation length")
    parser.add_argument("--model", default="random", type=str, choices=["random", "ppo", "q_learning", "sarsa", "dqn"], help="Agent model type")
    parser.add_argument("--gamma", default=0.95, type=float, help="Discount factor")
    parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate for Q-learning/Sarsa")
    parser.add_argument("--epsilon_decay", default=0.9997, type=float, help="Epsilon decay rate")
    parser.add_argument("--min_epsilon", default=0.05, type=float, help="Minimum epsilon")
    parser.add_argument("--entropy_coef", default=0.03, type=float, help="Entropy coefficient for PPO (controls stochastic exploration)")
    parser.add_argument("--hidden_layers", default="64,32", type=str, help="Comma-separated hidden layer sizes for NN-based models (DQN, PPO)")
    parser.add_argument("--slippery", action="store_true", default=True, help="Make the environment slippery")
    parser.add_argument("--log_dir", type=str, help="Custom directory for trajectories")
    args = parser.parse_args()
    
    # Force consistent episode counts based on slippery vs non-slippery baseline goals
    if args.episodes == 30000 or args.episodes == 5000:
        args.episodes = 30000 if args.slippery else 5000

    # Parse hidden layers
    args.hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    experiment_config = {
        "env": "FrozenLake-v1-8x8-slippery" if args.slippery else "FrozenLake-v1-8x8", # Custom name for logging
        "model": args.model,
        "gamma": args.gamma,
        "alpha": args.alpha, # Q-learning/Sarsa alpha
        "epsilon": 1.0,
        "epsilon_min": args.min_epsilon,
        "epsilon_decay": args.epsilon_decay,
        # Hyperparams for PPO/DQN
        "lr": 5e-4, 
        "clip_ratio": 0.2,
        "entropy_coef": args.entropy_coef,
        "batch_size": 32,
        "memory_size": 50000,
        "replay_each": 4,
        "target_update_every": 500,
        "hidden_layers": args.hidden_layers
    }
    experiment_config.update(vars(args))
    print(experiment_config)
    
    # basic env
    # FrozenLake-hard usually implies 8x8 map and slippery

    # FrozenLake-hard usually implies 8x8 map and slippery
    env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=args.slippery)
    env.action_space.seed(args.seed)
    # Resetting with seed ensures the environment's RNG is initialized. 
    # Subsequent resets during training (done by agent) will use this RNG state 
    # but produce different initial states if the env is stochastic or has random starts.
    # Note: FrozenLake v1 8x8 is always the same map start, but transitions are stochastic.
    env.reset(seed=args.seed)
    
    # Instantiate agent
    if args.model == "random":
        AgentClass = RandomAgent
    elif args.model == "ppo":
        AgentClass = PPOAgent
    elif args.model == "q_learning":
        AgentClass = QLearningAgent
    elif args.model == "sarsa":
        AgentClass = SarsaAgent
    elif args.model == "dqn":
        AgentClass = DQNAgent
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Pass everything in kwargs, but remove conflicting keys
    agent_kwargs = experiment_config.copy()
    agent_kwargs.pop("env", None)
    agent_kwargs.pop("model", None)
    
    # Filter args for log path to avoid filename too long
    common_keys = ["env", "model", "seed", "episodes", "evaluate_each", "evaluate_for", "slippery"]
    if args.model in ["q_learning", "sarsa"]:
        relevant_keys = common_keys + ["gamma", "alpha", "epsilon", "epsilon_min", "epsilon_decay"]
    elif args.model == "dqn":
        relevant_keys = common_keys + ["gamma", "epsilon", "epsilon_min", "epsilon_decay", "lr", "batch_size", "memory_size", "replay_each", "target_update_every", "hidden_layers"]
    elif args.model == "ppo":
        relevant_keys = common_keys + ["gamma", "lr", "clip_ratio", "entropy_coef", "hidden_layers"]
    else: # random
        relevant_keys = common_keys
        
    filtered_config = {k: experiment_config[k] for k in relevant_keys if k in experiment_config}
    filtered_config["mode"] = "standard"

    agent = AgentClass(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
        
    eval_results = agent.train_policy(env, num_episodes=experiment_config["episodes"], evaluate_each=experiment_config["evaluate_each"], evaluate_for=experiment_config["evaluate_for"])
    print(eval_results)

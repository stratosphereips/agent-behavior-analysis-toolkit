from environments.gym_discerete_wrappers import DiscreteMountainCarWrapper
from agents.random import RandomAgent
from agents.ppo import PPOAgent
from agents.dqn import DQNAgent
from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
import gymnasium as gym
import argparse
import numpy as np
import random
import tensorflow as tf

try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=4242, type=int, help="Random seed.")
    # Fixed at 10000 for MountainCar
    parser.add_argument("--episodes", default=10000, type=int, help="Number of training episodes")
    parser.add_argument("--evaluate_each", default=500, type=int, help="Periodic evluation frequency")
    parser.add_argument("--evaluate_for", default=200, type=int, help="Periodic evluation length")
    parser.add_argument("--model", default="random", type=str, choices=["random", "ppo", "dqn", "sarsa", "q_learning"], help="Agent model type")
    
    # Optional overrides
    parser.add_argument("--alpha", type=float, help="Learning rate (tabular)")
    parser.add_argument("--lr", type=float, help="Learning rate (NN)")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Epsilon (exploration)")
    parser.add_argument("--epsilon_min", type=float, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, help="Epsilon decay rate/factor")
    parser.add_argument("--entropy_coef", type=float, help="PPO entropy coefficient")
    parser.add_argument("--q_init_val", type=float, help="Initial Q table value")
    parser.add_argument("--log_dir", type=str, help="Custom directory for trajectories")
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    AGENT_CONFIGS = {
        "ppo": {
            "lr": 3e-4,
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "train_iters": 10,
            "batch_size": 64,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "hidden_layers": [64, 64]
        },
        "dqn": {
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9995, 
            "batch_size": 64,
            "memory_size": 20000,
            "replay_each": 4,
            "target_update_every": 1000,
             "hidden_layers": [64, 64]
        },
        "sarsa": {
            "alpha": 0.1, 
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9995,
            "q_init_val": 0.0
        },
        "q_learning": {
             "alpha": 0.1,
             "gamma": 0.99,
             "epsilon": 1.0,
             "epsilon_min": 0.01,
             "epsilon_decay": 0.9995,
             "q_init_val": 0.0
        },
        "random": {}
    }

    experiment_config = AGENT_CONFIGS[args.model].copy()
    experiment_config.update({"env": "MountainCar-v0-discrete", "model": args.model})
    experiment_config.update({k: v for k, v in vars(args).items() if v is not None})
    print("Running STANDARD MountainCar with config:")
    print(experiment_config)
    
    # Environment Setup
    env = gym.make("MountainCar-v0")
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    discretized_env = DiscreteMountainCarWrapper(env, bins=20)
    
    common_keys = ["env", "model", "seed", "episodes", "evaluate_each", "evaluate_for"]
    if args.model in ["q_learning", "sarsa"]:
        relevant_keys = common_keys + ["gamma", "alpha", "epsilon", "epsilon_min", "epsilon_decay", "q_init_val"]
    elif args.model == "dqn":
        relevant_keys = common_keys + ["gamma", "epsilon", "epsilon_min", "epsilon_decay", "lr", "batch_size", "memory_size", "replay_each", "target_update_every"]
    elif args.model == "ppo":
        relevant_keys = common_keys + ["gamma", "lr", "clip_ratio", "entropy_coef"]
    else:
        relevant_keys = common_keys
        
    filtered_config = {k: experiment_config[k] for k in relevant_keys if k in experiment_config}
    filtered_config["mode"] = "standard"
    
    agent_kwargs = experiment_config.copy()
    agent_kwargs.pop("env", None)
    agent_kwargs.pop("model", None)

    if args.model == "random":
        agent = RandomAgent(env=discretized_env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "ppo":
        agent = PPOAgent(env=discretized_env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "dqn":
        agent = DQNAgent(env=discretized_env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "sarsa":
        agent = SarsaAgent(env=discretized_env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "q_learning":
        agent = QLearningAgent(env=discretized_env, store_trajectories=True, args=filtered_config, **agent_kwargs)
        
    eval_results = agent.train_policy(discretized_env, num_episodes=experiment_config["episodes"], evaluate_each=experiment_config["evaluate_each"], evaluate_for=experiment_config["evaluate_for"])
    print(eval_results)

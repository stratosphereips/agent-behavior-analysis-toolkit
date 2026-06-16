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
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--alpha", type=float, help="Learning rate for Q-learning/Sarsa")
    parser.add_argument("--lr", type=float, help="Learning rate (NN)")
    parser.add_argument("--epsilon", type=float, help="Epsilon (exploration)")
    parser.add_argument("--epsilon_min", type=float, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, help="Epsilon decay rate")
    parser.add_argument("--entropy_coef", type=float, help="Entropy coefficient for PPO")
    parser.add_argument("--entropy_min", type=float, help="PPO entropy minimum")
    parser.add_argument("--entropy_decay_frac", type=float, help="Fraction of training over which entropy decays")
    parser.add_argument("--hidden_layers", type=int, nargs="+", help="Hidden layer sizes for NN-based models")
    parser.add_argument("--slippery", action="store_true", default=True, help="Make the environment slippery")
    parser.add_argument("--no_slippery", action="store_false", dest="slippery", help="Make the environment non-slippery")
    parser.add_argument("--q_init_val", type=float, help="Initial value for Q-table")
    parser.add_argument("--clip_ratio", type=float, help="PPO clip ratio")
    parser.add_argument("--lam", type=float, help="PPO lambda")
    parser.add_argument("--train_iters", type=int, help="PPO train iterations")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--value_coef", type=float, help="PPO value coefficient")
    parser.add_argument("--memory_size", type=int, help="DQN memory size")
    parser.add_argument("--replay_each", type=int, help="DQN replay frequency")
    parser.add_argument("--target_update_every", type=int, help="DQN target update frequency")
    parser.add_argument("--epsilon_decay_steps", type=int, help="DQN epsilon decay steps")
    parser.add_argument("--log_dir", type=str, help="Custom directory for trajectories")
    args = parser.parse_args()
    
    # Force consistent episode counts based on slippery vs non-slippery baseline goals
    if args.episodes == 30000 or args.episodes == 5000:
        args.episodes = 30000 if args.slippery else 5000

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Agent-specific configurations for FrozenLake
    AGENT_CONFIGS = {
        "ppo": {
            "lr": args.lr,
            "gamma": args.gamma,
            "clip_ratio": args.clip_ratio,
            "lam": args.lam,
            "train_iters": args.train_iters,
            "batch_size": args.batch_size,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "entropy_min": args.entropy_min,
            "entropy_decay_frac": args.entropy_decay_frac,
            "hidden_layers": args.hidden_layers
        },
        "dqn": {
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "batch_size": args.batch_size,
            "memory_size": args.memory_size,
            "replay_each": args.replay_each,
            "target_update_every": args.target_update_every,
            "hidden_layers": args.hidden_layers
        },
        "sarsa": {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "q_init_val": args.q_init_val
        },
        "q_learning": {
             "alpha": args.alpha,
             "gamma": args.gamma,
             "epsilon": args.epsilon,
             "epsilon_min": args.epsilon_min,
             "epsilon_decay": args.epsilon_decay,
             "q_init_val": args.q_init_val
        },
        "random": {}
    }

    # Start with the specific config for the selected model, filtering out unset (None) CLI args
    experiment_config = {k: v for k, v in AGENT_CONFIGS[args.model].items() if v is not None}
    
    # Add common/environment configs
    experiment_config.update({
        "env": "FrozenLake-v1-8x8-slippery" if args.slippery else "FrozenLake-v1-8x8",
        "model": args.model,
    })
    
    # Override with any CLI arguments that were explicitly set (not None)
    experiment_config.update({k: v for k, v in vars(args).items() if v is not None})
    print(f"Running {experiment_config['env']} with config:")
    print(experiment_config)
    

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
        relevant_keys = common_keys + ["gamma", "alpha", "epsilon", "epsilon_min", "epsilon_decay", "q_init_val"]
    elif args.model == "dqn":
        relevant_keys = common_keys + ["gamma", "epsilon", "epsilon_min", "epsilon_decay", "epsilon_decay_steps", "lr", "batch_size", "memory_size", "replay_each", "target_update_every", "hidden_layers"]
    elif args.model == "ppo":
        relevant_keys = common_keys + ["gamma", "lr", "clip_ratio", "entropy_coef", "entropy_min", "entropy_decay_frac", "hidden_layers"]
    else: # random
        relevant_keys = common_keys
        
    filtered_config = {k: experiment_config[k] for k in relevant_keys if k in experiment_config}
    filtered_config["mode"] = "standard"

    agent = AgentClass(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
        
    eval_results = agent.train_policy(env, num_episodes=experiment_config["episodes"], evaluate_each=experiment_config["evaluate_each"], evaluate_for=experiment_config["evaluate_for"])
    print(eval_results)

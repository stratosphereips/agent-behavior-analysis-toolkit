from agents.random import RandomAgent
from agents.ppo import PPOAgent
from agents.dqn import DQNAgent
from agents.sarsa import SarsaAgent
from agents.q_learning import QLearningAgent
import gymnasium as gym
import argparse
import numpy as np
import random

# Configure Tensorflow to use memory growth to allow multiple concurrent runs
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

class TaxiFeatureWrapper(gym.ObservationWrapper):
    """
    Translates the Taxi discrete state into a 7D perfect spatial representation.
    (taxi_row, taxi_col, pass_row, pass_col, dest_row, dest_col, in_taxi)
    This allows Neural Networks to easily learn Manhattan distances.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.locs = [(0,0), (0,4), (4,0), (4,3)]
        
    def observation(self, obs):
        taxi_r, taxi_c, pass_loc, dest_idx = list(self.env.unwrapped.decode(obs))
        
        in_taxi = 1.0 if pass_loc == 4 else 0.0
        
        if pass_loc == 4:
            pass_r, pass_c = taxi_r, taxi_c
        else:
            pass_r, pass_c = self.locs[pass_loc]
            
        dest_r, dest_c = self.locs[dest_idx]
        
        return np.array([
            taxi_r / 4.0,
            taxi_c / 4.0,
            pass_r / 4.0,
            pass_c / 4.0,
            dest_r / 4.0,
            dest_c / 4.0,
            in_taxi
        ], dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=4242, type=int, help="Random seed.")
    # Fixed at 3000 for all Taxi variants
    parser.add_argument("--episodes", default=3000, type=int, help="Number of training episodes")
    parser.add_argument("--evaluate_each", default=100, type=int, help="Periodic evluation frequency")
    parser.add_argument("--evaluate_for", default=200, type=int, help="Periodic evluation length")
    parser.add_argument("--model", default="random", type=str, choices=["random", "ppo", "dqn", "sarsa", "q_learning"], help="Agent model type")
    parser.add_argument("--env_type", default="both", type=str, choices=["standard", "fickle", "rainy", "both"], help="Environment type")
    
    # Hyperparameters (Optional overrides)
    parser.add_argument("--alpha", type=float, help="Learning rate (tabular)")
    parser.add_argument("--lr", type=float, help="Learning rate (NN)")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--epsilon", type=float, help="Epsilon (exploration)")
    parser.add_argument("--epsilon_min", type=float, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, help="Epsilon decay rate/factor")
    parser.add_argument("--entropy_coef", type=float, help="PPO entropy coefficient (initial value)")
    parser.add_argument("--entropy_min", type=float, help="PPO entropy minimum")
    parser.add_argument("--entropy_decay_frac", type=float, help="Fraction of training over which entropy decays (e.g. 0.3)")
    parser.add_argument("--q_init_val", type=float, help="Initial Q table value")
    parser.add_argument("--clip_ratio", type=float, help="PPO clip ratio")
    parser.add_argument("--lam", type=float, help="PPO lambda")
    parser.add_argument("--train_iters", type=int, help="PPO train iterations")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--value_coef", type=float, help="PPO value coefficient")
    parser.add_argument("--hidden_layers", type=int, nargs="+", help="Hidden layers")
    parser.add_argument("--epsilon_decay_steps", type=int, help="DQN epsilon decay steps")
    parser.add_argument("--memory_size", type=int, help="DQN memory size")
    parser.add_argument("--replay_each", type=int, help="DQN replay frequency")
    parser.add_argument("--target_update_every", type=int, help="DQN target update frequency")
    
    parser.add_argument("--log_dir", type=str, help="Custom directory for trajectories")
    
    args = parser.parse_args()

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Agent-specific configurations for Taxi (Standard)
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
    # so they don't override the agent's internal defaults
    experiment_config = {k: v for k, v in AGENT_CONFIGS[args.model].items() if v is not None}
    
    # Add common/environment configs
    experiment_config.update({
        "env": "Taxi-v3",
        "model": args.model,
    })
    
    # Override with any CLI arguments that were explicitly set (not None)
    experiment_config.update({k: v for k, v in vars(args).items() if v is not None})
    print("Running STANDARD Taxi-v3 with config:")
    print(experiment_config)
    
    # Create Env
    if args.env_type == "standard":
        env = gym.make("Taxi-v4")
    elif args.env_type == "fickle":
        env = gym.make("Taxi-v4",fickle_passenger=True, fickle_probability=0.1)
    elif args.env_type == "rainy":
        env = gym.make("Taxi-v4",is_rainy=True, rainy_probability=0.9)
    elif args.env_type == "both":
        env = gym.make("Taxi-v4",fickle_passenger=True, is_rainy=True, rainy_probability=0.9, fickle_probability=0.1)

    print(f"Running {args.env_type.upper()} Taxi-v3 with config:")
    print(experiment_config)
    
    # We do NOT use TaxiFeatureWrapper for PPO anymore.
    # Passing the raw discrete state allows PPO's internal one-hot encoding 
    # to treat this as a tabular problem, avoiding the need to learn maze walls from coordinates.
    
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    
    # Filter config for logging paths
    common_keys = ["env", "model", "seed", "episodes", "evaluate_each", "evaluate_for"]
    if args.model in ["q_learning", "sarsa"]:
        relevant_keys = common_keys + ["gamma", "alpha", "epsilon", "epsilon_min", "epsilon_decay", "q_init_val"]
    elif args.model == "dqn":
        relevant_keys = common_keys + ["gamma", "epsilon", "epsilon_min", "epsilon_decay", "lr", "batch_size", "memory_size", "replay_each", "target_update_every"]
    elif args.model == "ppo":
        relevant_keys = common_keys + ["gamma", "lr", "clip_ratio", "entropy_coef", "entropy_min", "entropy_decay_frac"]
    else:
        relevant_keys = common_keys
        
    filtered_config = {k: experiment_config[k] for k in relevant_keys if k in experiment_config}
    filtered_config["mode"] = "standard"

    agent_kwargs = experiment_config.copy()
    agent_kwargs.pop("env", None)
    agent_kwargs.pop("model", None)

    if args.model == "random":
        agent = RandomAgent(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "ppo":
        agent = PPOAgent(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "dqn":
        agent = DQNAgent(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "sarsa":
        agent = SarsaAgent(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
    elif args.model == "q_learning":
        agent = QLearningAgent(env=env, store_trajectories=True, args=filtered_config, **agent_kwargs)
        
    eval_results = agent.train_policy(env, num_episodes=experiment_config["episodes"], evaluate_each=experiment_config["evaluate_each"], evaluate_for=experiment_config["evaluate_for"])
    print(eval_results)

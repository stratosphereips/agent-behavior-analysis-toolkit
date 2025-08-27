
from environments.gym_wrappers import TrajectoryRecorderWrapper, CustomEnv
from agents.dqn import DQNAgent
from agents.reinforce import REINFORCE
import gymnasium as gym
import argparse
import numpy as np
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=4242, type=int, help="Random seed.")
    parser.add_argument("--episodes", default=10000, type=int, help="Number of training episodes")
    parser.add_argument("--evaluate_each", default=500, type=int, help="Periodic evluation frequency")
    parser.add_argument("--evaluate_for", default=500, type=int, help="Periodic evluation length")
    args = parser.parse_args()
    # Fix random seed
    np.random.seed(args.seed)
    # Define experiment configuration
            # Hyperparameters
    experiment_config = {
        "env": "CartPole-v1",
        "model": "DQN",
        "lr": 0.5,
        "gamma": 0.99,
        "epsilon": 1,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "memory_size": 10000
    }
    experiment_config.update(vars(args))
    print(experiment_config)
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="ondrej-lukas-czech-technical-university-in-prague",
        # Set the wandb project where this run will be logged.
        project="agent-trajectory-analysis",
        # Track hyperparameters and run metadata.
        config=experiment_config,
    ) 

    # basic env
    env = gym.make("CartPole-v1")
    # Wrap with custom seed logic for reproducibility
    env = CustomEnv(env, seed=args.seed)
    # add Trajectory recording Wrapper
    discretized_env = TrajectoryRecorderWrapper(env)
    # add the method which determines if an agent wins
    discretized_env.is_win_fn = lambda trajectory: len(trajectory) > 450
    
    match experiment_config["model"]:
        case "DQN":
            agent = DQNAgent(
                discretized_env.observation_space.shape,
                discretized_env.action_space.n,
                lr=experiment_config["lr"],
                gamma=experiment_config["gamma"],
                epsilon=experiment_config["epsilon"],
                epsilon_min=experiment_config["epsilon_min"],
                epsilon_decay=experiment_config["epsilon_decay"],
                wandb_run=wandb_run,
                experiment_config=experiment_config
        )

    agent.train_policy(discretized_env, num_episodes=args.episodes, evaluate_each=args.evaluate_each, evaluate_for=args.evaluate_for)
    # Final evaluation
    returns, _ = agent.evaluate_policy(discretized_env, num_episodes=2000,final_evaluation=True)
    print(f"Final evaluation:{np.mean(returns):.2f}+-{np.std(returns):.2f}")
    wandb_run.finish()
    
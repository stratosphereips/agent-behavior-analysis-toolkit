from environments.gym_discerete_wrappers import DiscreteCartPoleWrapper
from environments.gym_wrappers import TrajectoryRecorderWrapper, CustomEnv
from agents.q_learning import Qlearning
from agents.sarsa import Sarsa
from agents.dqn import DQN
from agents.random import RandomAgent
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
    experiment_config = {
        "env": "CartPole-v1-discrete",
        "model": "Random",
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
    # Add Discretization Layer
    discretized_env = DiscreteCartPoleWrapper(env, bins=8)
    # add Trajectory recording Wrapper
    discretized_env = TrajectoryRecorderWrapper(discretized_env)
    # add the method which determines if an agent wins
    discretized_env.is_win_fn = lambda trajectory: len(trajectory) > 450
    
    agent = Qlearning(
        discretized_env.observation_space.n,
        discretized_env.action_space.n,
        alpha=0.1,
        gamma=0.95,
        epsilon=1,
        epsilon_min=0.01,
        epsilon_decay=0.0001,
        wandb_run = wandb_run
        )

    #agent = RandomAgent(discretized_env.action_space.n, wandb_run = wandb_run)

    # agent = Sarsa(
    #     discretized_env.observation_space.n,
    #     discretized_env.action_space.n,
    #     alpha=0.1,
    #     gamma=0.95,
    #     epsilon=1,
    #     epsilon_min=0.05,
    #     epsilon_decay=0.0005,
    #     wandb_run = wandb_run
    # )
    print(discretized_env.action_space.shape)  


    agent.train_policy(discretized_env, num_episodes=args.episodes, evaluate_each=args.evaluate_each, evaluate_for=args.evaluate_for)
    # Final evaluation
    returns, _ = agent.evaluate_policy(discretized_env, num_episodes=2000,final_evaluation=True)
    print(f"Final evaluation:{np.mean(returns):.2f}+-{np.std(returns):.2f}")
    wandb_run.finish()
    
from environments.cartpole_discrete import DiscretizedCartPoleEnv
from agents.q_learning import Qlearning




if __name__ == "__main__":
    discretized_env = DiscretizedCartPoleEnv(num_bins=10)
    agent = Qlearning(
        discretized_env.observation_space_size,
        discretized_env.action_space.n,
        alpha=0.1,
        gamma=0.95,
        epsilon=1,
        epsilon_min=0.01,
        epsilon_decay=0.995
        )
    agent.train_policy(discretized_env, num_episodes=10000, evaluate_each=1000, evaluate_for=1000)
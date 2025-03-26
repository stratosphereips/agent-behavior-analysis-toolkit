import numpy as np
from .base_agent import Agent
import math

def epsilon_greedy(Q, state, epsilon, action_space):
    """Select an action using the epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    return np.argmax(Q[state])

def exponential_epsilon_decay(episode, initial_epsilon, min_epsilon, decay_rate):
    """Calculates epsilon using exponential decay based on episode number."""
    return max(min_epsilon, initial_epsilon * math.exp(-decay_rate * episode))

class Qlearning(Agent):
    
    def __init__(self,obs_space_size, acion_space_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = np.zeros((obs_space_size, acion_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space_size = acion_space_size

    def step(self, state, training=False):
        if training:
            return epsilon_greedy(self.Q, state, self.epsilon, self.action_space_size)
        else:
            return epsilon_greedy(self.Q, state, 0,self.action_space_size)
    
    def train_policy(self, env, num_episodes:int, evaluate_each:int, evaluate_for:int):
        """Q_learning algorithm with periodic evaluation."""
        eval_results = []  # Store evaluation results
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # update rule
                if not done:
                    self.Q[state, action] += self.alpha * (reward + self.gamma * max(self.Q[next_state, :]) - self.Q[state, action])
                else:
                    self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
                state = next_state  # Move to the next step
            
            # Decay epsilon (gradually reduce exploration)
            self.epsilon = exponential_epsilon_decay(episode, 1.0, self.epsilon_min, self.epsilon_decay)

            # Periodic evaluation
            if evaluate_each:
                if episode % evaluate_each == 0:
                    returns,_ = self.evaluate_policy(env, evaluate_for)
                    mean_ret =  np.mean(returns)
                    eval_results.append((episode, mean_ret))
                    print(f"Episode {episode}: Mean return = {mean_ret:.2f}")
        return eval_results
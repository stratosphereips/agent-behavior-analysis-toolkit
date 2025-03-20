import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
from .base_agent import Agent

class REINFORCE(Agent):
    def __init__(self, obs_space_size: int, action_space_size: int, learning_rate: float = 0.001, gamma: float = 0.99):
        super().__init__()
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Store episodes (state, action, reward)
        self.episode_memory = []

        # Policy Network
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Build a policy network with softmax output."""
        model = Sequential([
            Dense(128, input_dim=self.obs_space_size, activation='relu'),
            Dropout(0.2),  # Dropout to help prevent overfitting
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),  # Deeper layer with fewer neurons
            Dense(self.action_space_size, activation='softmax')  # Output is probability distribution
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def remember(self, state, action, reward):
        """Store experience in episode memory."""
        self.episode_memory.append((state, action, reward))

    def step(self, state) -> int:
        """Select an action using the policy network."""
        state = np.reshape(state, [1, self.obs_space_size])
        action_probs = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space_size, p=action_probs)  # Sample action
        return action

    def train(self):
        """Train policy using REINFORCE algorithm."""
        states, actions, rewards = zip(*self.episode_memory)

        # Compute discounted returns (G)
        G = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative

        # Convert data to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(G, dtype=np.float32)

        # Convert actions to one-hot encoding
        action_masks = np.zeros((len(actions), self.action_space_size))
        action_masks[np.arange(len(actions)), actions] = 1

        # Train the policy network
        self.model.fit(states, action_masks, sample_weight=returns, epochs=1, verbose=0)

        # Clear memory after training
        self.episode_memory = []

    def train_policy(self, env, num_episodes: int, evaluate_each: int, evaluate_for: int) -> list:
        """Train REINFORCE agent and evaluate periodically."""
        eval_results = []
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.obs_space_size])
            done = False
            while not done:
                action = self.step(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.obs_space_size])
                self.remember(state[0], action, reward)
                state = next_state

            # Train policy after every episode
            self.train()

            # Evaluate policy periodically
            if evaluate_each and episode % evaluate_each == 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                mean_ret = np.mean(returns)
                eval_results.append((episode, mean_ret))
                print(f"Episode {episode}: Mean return = {mean_ret:.2f}")

        return eval_results
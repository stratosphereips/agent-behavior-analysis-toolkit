import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from collections import deque
from .base_agent import Agent

class REINFORCE(Agent):
    def __init__(self, obs_space_size: int, action_space_size: int, learning_rate: float = 0.0005, gamma: float = 0.99):
        super().__init__()
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # More efficient storage for episode memory
        self.episode_memory = deque(maxlen=5000)

        # Policy Network
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Build a policy network with softmax output."""
        model = Sequential([
            Dense(128, input_dim=self.obs_space_size),
            LeakyReLU(alpha=0.01),  # Smoother gradients
            Dense(128),
            LeakyReLU(alpha=0.01),
            Dense(64),
            LeakyReLU(alpha=0.01),
            Dense(self.action_space_size, activation='softmax')  # Probability distribution
        ])
        model.compile(optimizer=RMSprop(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def remember(self, state, action, reward):
        """Store experience in episode memory."""
        self.episode_memory.append((state, action, reward))

    def step(self, state) -> int:
        """Select an action using the policy network."""
        state = np.reshape(state, [1, self.obs_space_size])
        action_probs = self.model.predict(state, verbose=0)[0]
        return np.random.choice(self.action_space_size, p=action_probs)

    def train(self):
        """Train policy using REINFORCE algorithm."""
        if not self.episode_memory:
            return

        states, actions, rewards = zip(*self.episode_memory)
        states = np.array(states, dtype=np.float32)

        # Compute discounted returns
        G = np.zeros(len(rewards), dtype=np.float32)    
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative

        # Normalize returns to stabilize training
        G = (G - np.mean(G)) / (np.std(G) + 1e-8)

        # One-hot encode actions
        action_masks = np.eye(self.action_space_size)[np.array(actions, dtype=np.int32)]

        # Train policy network
        self.model.fit(states, action_masks, sample_weight=G, epochs=1, verbose=0)

        # Clear memory after training
        self.episode_memory.clear()

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
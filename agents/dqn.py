import numpy as np
import random
from typing import List, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from .base_agent import Agent
import tensorflow as tf

class DQN(Agent):
    def __init__(self, obs_space_size: int, action_space_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 32, memory_size: int = 100000,
                 hidden_units: int = 128, train_each=4):
        super().__init__()
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.train_each = train_each

        # Memory for experience replay (state, action, reward, next_state, done)
        self.memory = deque(maxlen=self.memory_size)
        self.memory_counter = 0

        # Create models
        self.model = self._build_model(hidden_units)
        self.target_model = self._build_model(hidden_units)
        self.update_target_model()

    def _build_model(self, hidden_units: int) -> Sequential:
        model = Sequential([
            Dense(hidden_units, input_dim=self.obs_space_size, activation='relu'),
            Dense(hidden_units, activation='relu'),
            Dense(self.action_space_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self, tau=0.01):
        """Soft update target model weights"""
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [tau * mw + (1 - tau) * tw for mw, tw in zip(model_weights, target_weights)]
        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def step(self, state, training=False):
        """ Choose an action based on epsilon-greedy policy. """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)      
        
        # Ensure state is properly shaped for prediction
        state = np.reshape(state, [1, self.obs_space_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """ Train the model using experience replay. """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory yet

        # Sample a batch from the deque (random.sample is faster and avoids conversion overhead)
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        
        # Unpack batch elements into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert lists to NumPy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Predict current Q-values
        targets = self.model.predict(states, verbose=0)

        # Predict next-state Q-values using target model
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values for the actions taken
        target_values = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)
        targets[np.arange(self.batch_size), actions] = target_values

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

    def train_policy(self, env, num_episodes: int, evaluate_each: int, evaluate_for: int, target_update_interval: int = 10) -> List[Tuple[int, float]]:
        """ Train the DQN agent in the environment. """
        eval_results = []
        step_counter = 0 
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.obs_space_size])
            done = False
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.obs_space_size])
                self.remember(state[0], action, reward, next_state[0], done)
                step_counter += 1
                if step_counter % self.train_each == 0 and len(self.memory) > self.batch_size:
                    self.replay()
                    # Linear decay of epsilon
                    self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min) / (num_episodes * 0.5))
                state = next_state
            
            # Update target network periodically
            if episode % target_update_interval == 0:
                self.update_target_model()

            # Evaluate policy periodically
            if evaluate_each and episode % evaluate_each == 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                mean_ret = np.mean(returns)
                eval_results.append((episode, mean_ret))
                print(f"Episode {episode}: Mean return = {mean_ret:.2f}")

        return eval_results
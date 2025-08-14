import numpy as np
import random
from typing import List, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import AdamW
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

        # Create models
        self._loss_fn = tf.keras.losses.Huber()
        self._optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=1e-5)
        self.model = self._build_model(hidden_units)
        self.model.compile(loss=self._loss_fn, optimizer=self._optimizer)
        self.target_model = self._build_model(hidden_units)
        self.update_target_model()

    def _build_model(self, hidden_units: int) -> Sequential:
        model = Sequential([
            Dense(hidden_units, input_dim=self.obs_space_size, activation='relu'),
            Dense(hidden_units, activation='relu'),
            Dense(self.action_space_size, activation='linear')
        ])
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
        """Choose an action based on epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return int(tf.random.categorical(tf.math.log([[1.0] * self.action_space_size]), 1)[0, 0])
        state = np.reshape(state, [1, self.obs_space_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def sample_batch(self):
        """Sample a batch from the memory using Python/NumPy."""
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32)
        )

    @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # states
        tf.TensorSpec(shape=[None], dtype=tf.int32),  # actions
        tf.TensorSpec(shape=[None], dtype=tf.float32),  # rewards
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # next_states
        tf.TensorSpec(shape=[None], dtype=tf.float32)  # dones
    ]
    )
    def train_step(self, states, actions, rewards, next_states, dones):
        """Optimized training step with TensorFlow graph execution."""
        with tf.GradientTape() as tape:
            current_q = self.model(states)  
            next_q = self.target_model(next_states)  
            target_values = rewards + self.gamma * tf.reduce_max(next_q, axis=1) * (1.0 - dones)
            target_values = tf.stop_gradient(tf.reshape(target_values, [-1]))  

            # Select action Q-values efficiently
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            predicted_q_values = tf.gather_nd(current_q, action_indices)

            loss = self._loss_fn(target_values, predicted_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def replay(self):
        """Train the model using experience replay."""
        batch = self.sample_batch()
        if batch is None:
            return  # Not enough samples in memory yet

        states, actions, rewards, next_states, dones = batch

        # Call the TF graph training step.
        loss = self.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_policy(self, env, num_episodes: int, evaluate_each: int, evaluate_for: int, target_update_interval: int = 10) -> List[Tuple[int, float]]:
        """Train the DQN agent in the environment."""
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

            # Evaluate policy periodically (evaluation code unchanged)
            if evaluate_each and episode % evaluate_each == 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                mean_ret = tf.math.reduce_mean(returns)
                eval_results.append((episode, mean_ret))
                print(f"Episode {episode}: Mean return = {mean_ret:.2f}")

        return eval_results
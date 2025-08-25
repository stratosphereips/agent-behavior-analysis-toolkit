import numpy as np
import tensorflow as tf
from collections import deque
import random
from agents.base_agent import Agent


class DQNAgent(Agent):

    def __init__(self, obs_space, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_space = obs_space
        self.action_space = action_space
        self.params = kwargs
        # Hyperparameters
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay = self.params.get("epsilon_decay", 0.999)
        self.lr = self.params.get("lr", 0.001)
        self.batch_size = self.params.get("batch_size", 64)
        self.memory_size = self.params.get("memory_size", 10000)
        self.replay_each = self.params.get("replay_each", 8)
        self.target_update_every = self.params.get("target_update_every", 500)

        # Replay buffer
        self.memory = deque(maxlen=self.memory_size)

        # Build Q-networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.Huber()
    
    def _build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.obs_space)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.action_space, activation="linear"),
        ])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Only train if enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # Convert to tensors
        states      = np.array(states)
        next_states = np.array(next_states)
        actions     = actions.astype(int)
        rewards     = rewards.astype(float)
        dones       = dones.astype(bool)

        # Predict current Q-values
        q_values = self.q_network(states, training=True)

        # Predict next-state Q-values (from target network!)
        next_q_values = self.target_network(next_states, training=False)
        max_next_q = np.max(next_q_values.numpy(), axis=1)

        # Build target Q-values
        target_q = q_values.numpy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # Train step
        with tf.GradientTape() as tape:
            q_pred = self.q_network(states, training=True)
            loss = self.loss_fn(target_q, q_pred)
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def act(self, state, training=False):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.q_network(np.array([state]))
        return np.argmax(q_values[0].numpy())

    def step(self, state, training=False):
        action = self.act(state, training=training)
        return action

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        steps = 0
        for ep in range(num_episodes):
            state, _ = env.reset()
            done, truncated = False, False
            total_reward = 0

            while not (done or truncated):
                steps += 1
                action = self.step(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

                # Store transition
                self.remember(state, action, reward, next_state, done or truncated)

                # Train if memory is ready
                if len(self.memory) > self.batch_size and steps % self.replay_each == 0:
                    self.replay()

                # Sync target network periodically
                if steps % self.target_update_every == 0:
                    self.target_network.set_weights(self.q_network.get_weights())

                state = next_state

            # Evaluate periodically
            if evaluate_each and ep % evaluate_each == 0 and ep > 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                mean_ret = np.mean(returns)
                print(f"Episode {ep}: TrainReturn={total_reward:.2f}, EvalReturn={mean_ret:.2f}")

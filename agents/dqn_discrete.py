import numpy as np
import tensorflow as tf
from collections import deque
import random
from .base_agent import Agent

class DQNDiscreteAgent(Agent):
    """
    A Deep Q-Network (DQN) agent adapted for the Taxi-v3 environment.
    """

    def __init__(self, obs_space, action_space, *args, **kwargs):
        """
        Initializes the DQN agent with hyperparameters and neural networks.

        Args:
            obs_space (int): The size of the discrete observation space.
            action_space (int): The size of the action space.
        """
        super().__init__(*args, **kwargs)
        self.obs_space = obs_space
        self.action_space = action_space
        self.params = kwargs

        # Hyperparameters for the DQN algorithm
        self.gamma = self.params.get("gamma", 0.99)
        self.epsilon = self.params.get("epsilon", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        # Slower decay rate to encourage more exploration
        self.epsilon_decay = self.params.get("epsilon_decay", 0.999)
        self.lr = self.params.get("lr", 0.001)
        self.batch_size = self.params.get("batch_size", 64)
        self.memory_size = self.params.get("memory_size", 10000)
        self.replay_each = self.params.get("replay_each", 10)
        self.target_update_every = self.params.get("target_update_every", 100)

        # Replay buffer to store experiences
        self.memory = deque(maxlen=self.memory_size)

        # Build and initialize the Q-network and target network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        # Copy initial weights from Q-network to target network
        self.target_network.set_weights(self.q_network.get_weights())

        # Optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.Huber()
    
    def _build_network(self):
        """
        Creates the neural network model for the Q-value approximation.
        The input layer is designed to handle a discrete state as a single integer.
        """
        model = tf.keras.Sequential([
            # Input layer for a single integer state
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Embedding(input_dim=500, output_dim=128),
            # Flatten the output of the embedding layer
            tf.keras.layers.Flatten(),
            # Use a dense layer to process the integer state
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            # Output layer for the 6 discrete actions
            tf.keras.layers.Dense(self.action_space, activation="linear"),
        ])
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Performs a single training step on a random batch from the replay buffer.
        This is where the main improvements were made for efficiency.
        """
        # Only train if enough samples are in the memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a random minibatch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Unpack the minibatch and convert to NumPy arrays.
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # --- Performance Improvement ---
        # Convert all NumPy arrays to TensorFlow tensors once at the beginning.
        # Reshape states to a 2D tensor (batch_size, 1) for the network input.
        states = tf.convert_to_tensor(np.array(states), dtype=tf.int32)
        states = tf.reshape(states, [-1, 1])
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.int32)
        next_states = tf.reshape(next_states, [-1, 1])
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        # Predict next-state Q-values from the target network
        next_q_values = self.target_network(next_states, training=False)
        # Find the maximum Q-value for each next state
        max_next_q = tf.reduce_max(next_q_values, axis=1)

        # Vectorized calculation of the target Q-values.
        # Target Q = reward + gamma * max(Q_next) * (1 - done)
        target_q_values = rewards + self.gamma * max_next_q * (1 - tf.cast(dones, tf.float32))

        # Predict current Q-values from the Q-network
        with tf.GradientTape() as tape:
            q_pred_all = self.q_network(states, training=True)
            # Gather the predicted Q-values for the actions that were taken
            q_pred = tf.gather_nd(q_pred_all, tf.stack([tf.range(self.batch_size), actions], axis=1))
            # Calculate the loss between the target Q-values and the predicted Q-values
            loss = self.loss_fn(target_q_values, q_pred)

        # Compute gradients and apply them to update the Q-network weights
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def act(self, state, training):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
            state (int): The current state from the Taxi-v3 environment.
            training (bool): If True, uses epsilon-greedy. If False, uses greedy.
        
        Returns:
            int: The chosen action.
        """
        # Epsilon-greedy exploration
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)

        # The state is a single integer, so we wrap it in a list to create a batch of 1
        state_tensor = tf.convert_to_tensor([state], dtype=tf.int32)
        q_values = self.q_network(state_tensor, training=False)
        return tf.argmax(q_values[0]).numpy()

    def step(self, state, training):
        """
        Alias for the act method.
        """
        return self.act(state, training=training)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        """
        Main training loop for the agent.
        
        Args:
            env: The environment to train on.
            num_episodes (int): The number of episodes to train for.
            evaluate_each (int): The frequency (in episodes) to evaluate the policy.
            evaluate_for (int): The number of episodes to run for evaluation.
        """
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

                # Store the experience tuple in the replay buffer
                self.remember(state, action, reward, next_state, done or truncated)

                # Train the network if enough experiences are collected
                if len(self.memory) > self.batch_size and steps % self.replay_each == 0:
                    self.replay()

                # Sync the target network with the Q-network periodically
                if steps % self.target_update_every == 0:
                    self.target_network.set_weights(self.q_network.get_weights())

                state = next_state

            # Evaluate the agent periodically and print progress
            if evaluate_each and ep % evaluate_each == 0 and ep > 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                mean_ret = np.mean(returns)
                print(f"Episode {ep}: TrainReturn={total_reward:.2f}, EvalReturn={mean_ret:.2f}")
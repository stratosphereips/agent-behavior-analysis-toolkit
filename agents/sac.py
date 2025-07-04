import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import random
from base_agent import Agent
from collections import deque

# Set float precision
tf.keras.backend.set_floatx('float32')

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards.reshape(-1,1), next_states, dones.reshape(-1,1)

    def __len__(self):
        return len(self.buffer)

# Gaussian policy network with reparameterization trick
class GaussianPolicy(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden1 = layers.Dense(hidden_dim, activation='relu')
        self.hidden2 = layers.Dense(hidden_dim, activation='relu')
        self.mean_layer = layers.Dense(act_dim)
        self.log_std_layer = layers.Dense(act_dim)

    def call(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


# Q network
class QNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.hidden1 = layers.Dense(hidden_dim, activation='relu')
        self.hidden2 = layers.Dense(hidden_dim, activation='relu')
        self.q_out = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        q = self.q_out(x)
        return q

class SACAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]
        self.gamma = self.params.get("gamma", 0.99)
        self.tau = self.params.get("tau", 0.005)
        self.batch_size = self.params.get("batch_size", 256)
        self.lr = self.params.get("lr", 3e-4)
        self.alpha = tf.Variable(self.params.get("alpha", 0.2), dtype=tf.float32)
        self.target_entropy = -env.action_space.shape[0]
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha_optimizer = optimizers.Adam(self.lr)

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Define networks
        self.policy = GaussianPolicy(self.obs_dim, self.act_dim)
        self.q1 = QNetwork(self.obs_dim, self.act_dim)
        self.q2 = QNetwork(self.obs_dim, self.act_dim)
        self.q1_target = QNetwork(self.obs_dim, self.act_dim)
        self.q2_target = QNetwork(self.obs_dim, self.act_dim)
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

        # Optimizers
        self.policy_optimizer = optimizers.Adam(self.lr)
        self.q1_optimizer = optimizers.Adam(self.lr)
        self.q2_optimizer = optimizers.Adam(self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.params.get("buffer_size", 1_000_000))

    def step(self, state, training=True):
        state = np.array(state, dtype=np.float32)
        if training:
            action = self.get_action(state)
        else:
            action = self.get_action(state, evaluate=True)
        return action

    def get_action(self, state, evaluate=False):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        if evaluate:
            mean, _ = self.policy(state_tensor)
            action = tf.tanh(mean)
            return action.numpy()[0]
        else:
            action, _ = self.policy.sample(state_tensor)
            return action.numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Q networks
        with tf.GradientTape(persistent=True) as tape:
            next_actions, next_log_pi = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = tf.minimum(target_q1, target_q2) - tf.exp(self.log_alpha) * next_log_pi
            target = rewards + (1 - dones) * self.gamma * target_q

            q1_pred = self.q1(states, actions)
            q2_pred = self.q2(states, actions)
            q1_loss = tf.reduce_mean((q1_pred - target) ** 2)
            q2_loss = tf.reduce_mean((q2_pred - target) ** 2)

        q1_grads = tape.gradient(q1_loss, self.q1.trainable_variables)
        q2_grads = tape.gradient(q2_loss, self.q2.trainable_variables)
        self.q1_optimizer.apply_gradients(zip(q1_grads, self.q1.trainable_variables))
        self.q2_optimizer.apply_gradients(zip(q2_grads, self.q2.trainable_variables))
        del tape

        # Update policy
        with tf.GradientTape() as tape:
            new_actions, log_pi = self.policy.sample(states)
            q1_new = self.q1(states, new_actions)
            q2_new = self.q2(states, new_actions)
            min_q_new = tf.minimum(q1_new, q2_new)
            policy_loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_pi - min_q_new)

        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        # Update alpha
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy))
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # Soft update target networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    def _soft_update(self, source, target):
        for src_var, tgt_var in zip(source.variables, target.variables):
            tgt_var.assign(self.tau * src_var + (1 - self.tau) * tgt_var)

    def train_policy(self, env, num_episodes, evaluate_each=None, evaluate_for=None):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.step(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                self.update()
            total_rewards.append(episode_reward)

            if evaluate_each and (episode + 1) % evaluate_each == 0:
                returns, _ = self.evaluate_policy(env, evaluate_for)
                print(f"Evaluation after episode {episode+1}: mean return = {np.mean(returns):.2f}")

        return total_rewards
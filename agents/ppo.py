import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sac import GaussianPolicy
import numpy as np
from base_agent import Agent

class PPOAgent(Agent):
    def _initialize_agent(self):
        env = self.params["env"]
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.clip_ratio = self.params.get("clip_ratio", 0.2)
        self.gamma = self.params.get("gamma", 0.99)
        self.lam = self.params.get("lam", 0.95)  # GAE lambda
        self.train_iters = self.params.get("train_iters", 80)
        self.batch_size = self.params.get("batch_size", 64)
        self.lr = self.params.get("lr", 3e-4)

        # Policy network outputs mean and log_std (Gaussian policy)
        self.policy = GaussianPolicy(self.obs_dim, self.act_dim)
        self.policy_old = GaussianPolicy(self.obs_dim, self.act_dim)
        self.policy_old.set_weights(self.policy.get_weights())

        # Value network for state value estimation
        self.value_net = self._build_value_net()
        self.value_optimizer = optimizers.Adam(self.lr)
        self.policy_optimizer = optimizers.Adam(self.lr)

        self.replay_buffer = []  # Will hold rollout trajectories

    def _build_value_net(self):
        inputs = layers.Input(shape=(self.obs_dim,))
        x = layers.Dense(64, activation='tanh')(inputs)
        x = layers.Dense(64, activation='tanh')(x)
        out = layers.Dense(1)(x)
        model = tf.keras.Model(inputs, out)
        return model

    def step(self, state, training=False):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        mean, log_std = self.policy_old(state)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)
        if training:
            action = dist.sample()
        else:
            action = mean
        action = tf.tanh(action)
        return action.numpy()[0]

    def store_transition(self, transition):
        # Transition = (state, action, reward, next_state, done, logp, value)
        self.replay_buffer.append(transition)

    def compute_gae(self, rewards, values, dones, next_values):
        advantages = []
        gae = 0
        values = np.append(values, next_values)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def update(self):
        # Convert replay buffer to arrays
        states = np.array([t[0] for t in self.replay_buffer])
        actions = np.array([t[1] for t in self.replay_buffer])
        rewards = np.array([t[2] for t in self.replay_buffer])
        next_states = np.array([t[3] for t in self.replay_buffer])
        dones = np.array([t[4] for t in self.replay_buffer])
        old_logps = np.array([t[5] for t in self.replay_buffer])
        values = np.array([t[6] for t in self.replay_buffer])

        # Compute next values for GAE
        next_values = self.value_net(next_states).numpy().squeeze()
        advantages = self.compute_gae(rewards, values, dones, next_values)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_logps, returns, advantages))
        dataset = dataset.shuffle(buffer_size=len(states)).batch(self.batch_size)

        for _ in range(self.train_iters):
            for batch_states, batch_actions, batch_old_logps, batch_returns, batch_advantages in dataset:
                self._train_step(batch_states, batch_actions, batch_old_logps, batch_returns, batch_advantages)

        # Update old policy weights
        self.policy_old.set_weights(self.policy.get_weights())
        self.replay_buffer = []

    @tf.function
    def _train_step(self, states, actions, old_logps, returns, advantages):
        with tf.GradientTape(persistent=True) as tape:
            # Policy loss
            mean, log_std = self.policy(states)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            logps = tf.reduce_sum(dist.log_prob(actions), axis=1)
            ratio = tf.exp(logps - old_logps)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Value loss
            values = self.value_net(states)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)

            # Entropy bonus (optional)
            entropy = tf.reduce_mean(dist.entropy())
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        policy_grads = tape.gradient(total_loss, self.policy.trainable_variables)
        value_grads = tape.gradient(total_loss, self.value_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_variables))
        del tape

    def train_policy(self, env, num_episodes, rollout_length=2048):
        for episode in range(num_episodes):
            state, _ = env.reset()
            ep_reward = 0
            for _ in range(rollout_length):
                state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
                mean, log_std = self.policy_old(state_tensor)
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mean, std)
                action = dist.sample()
                tanh_action = tf.tanh(action).numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(tanh_action)
                done = terminated or truncated

                logp = tf.reduce_sum(dist.log_prob(action), axis=1).numpy()[0]
                value = self.value_net(state_tensor).numpy()[0,0]

                self.store_transition((state, tanh_action, reward, next_state, done, logp, value))

                state = next_state
                ep_reward += reward

                if done:
                    state, _ = env.reset()

            self.update()
            print(f"Episode {episode+1}, Reward: {ep_reward:.2f}")

import gymnasium as gym
import numpy as np
import os 
import sys

class EvaluationEnv(gym.Wrapper):
    def __init__(self, env, seed=None, evaluate_for=100, report_each=10):
        super().__init__(env)
        self._evaluate_for = evaluate_for
        self._report_each = report_each
        self._report_verbose = os.getenv("VERBOSE") not in [None, "", "0"]
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self._episode_running = False
        self._episode_returns = []
        self._evaluating_from = None
        self.seed = seed

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, start_evaluation=False, logging=True):
        if self._evaluating_from is not None and self._episode_running:
            raise RuntimeError("Cannot reset a running episode after `start_evaluation=True`")

        if start_evaluation and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_running = True
        self._episode_return = 0 if logging or self._evaluating_from is not None else None
        return super().reset()

    def step(self, action):
        if not self._episode_running:
            raise RuntimeError("Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = super().step(action)

        self._episode_running = not done
        if self._episode_return is not None:
            self._episode_return += reward
        if self._episode_return is not None and done:
            self._episode_returns.append(self._episode_return)

            if self._report_each and self.episode % self._report_each == 0:
                print("Episode {}, mean {}-episode return {:.2f} +-{:.2f}{}".format(
                    self.episode, self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:]), "" if not self._report_verbose else
                    ", returns " + " ".join(map("{:g}".format, self._episode_returns[-self._report_each:]))),
                    file=sys.stderr, flush=True)
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} +-{:.2f}".format(
                    self._evaluate_for, np.mean(self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), flush=True)
                self.close()
                sys.exit(0)

        return observation, reward, done, info
            
class DiscretizationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env, separators, tiles=None):
        super().__init__(env)
        self._separators = separators
        self._tiles = tiles

        if tiles is None:
            states = 1
            for separator in separators:
                states *= 1 + len(separator)
            self.observation_space = gym.spaces.Discrete(states)
        else:
            self._first_tile_states, self._rest_tiles_states = 1, 1
            for separator in separators:
                self._first_tile_states *= 1 + len(separator)
                self._rest_tiles_states *= 2 + len(separator)
            self.observation_space = gym.spaces.MultiDiscrete([
                self._first_tile_states + i * self._rest_tiles_states for i in range(tiles)])

            self._separator_offsets, self._separator_tops = [], []
            for separator in separators:
                self._separator_offsets.append(0 if len(separator) <= 1 else (separator[1] - separator[0]) / tiles)
                self._separator_tops.append(np.inf if len(separator) <= 1 else separator[-1] + (separator[1] - separator[0]))
    
    def observation(self, observations):
        state = 0
        for observation, separator in zip(observations, self._separators):
            state *= 1 + len(separator)
            state += np.digitize(observation, separator)
        if self._tiles is None:
            return state
        else:
            states = [state]
            for t in range(1, self._tiles):
                state = 0
                for i in range(len(self._separators)):
                    state *= 2 + len(self._separators[i])
                    value = observations[i] + ((t * (2 * i + 1)) % self._tiles) * self._separator_offsets[i]
                    if value > self._separator_tops[i]:
                        state += 1 + len(self._separators[i])
                    else:
                        state += np.digitize(value, self._separators[i])
                states.append(self._first_tile_states + (t - 1) * self._rest_tiles_states + state)
            return states
    
    # Expose the Recording API
    def start_recording(self):
        return getattr(self.env, "start_recording")()

    def stop_recording(self):
        return getattr(self.env, "stop_recording")()

    @property
    def trajectory_log(self):
        return getattr(self.env, "trajectory_log")
    
    @property
    def current_trajectory(self):
        return getattr(self.env, "current_trajectory")
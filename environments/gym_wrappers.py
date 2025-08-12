from trajectory import Trajectory, Transition
import gymnasium as gym
import numpy as np
import copy


class CustomEnv(gym.Wrapper):
    """
    Gymnasium wrapper that enables fixing random seed.
    """
    def __init__(self, env, seed=None):
        super().__init__(env)
        self.seed = seed
    
    def reset(self, seed=None, options=None):
        if self.seed is not None:
            return super().reset(seed=self.seed, options=options)
        else:
            return super().reset(seed=seed, options=options)

class TrajectoryRecorderWrapper(gym.Wrapper):
    """
    Wrapper that enables to record trajectories of the agnts
    """
    def __init__(self, env):
        super().__init__(env)
        self.record_trajectory = False
        self.trajectory_log = []         # List of full episode transitions
        self.current_trajectory = None
        self._last_real_obs = None

    def start_recording(self):
        self.record_trajectory = True
        self.current_trajectory = Trajectory()

    def stop_recording(self):
        self.record_trajectory = False
    
    def clear_trajectory_log(self)->None:
        self.trajectory_log = []

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Unpack if it's (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        if self.record_trajectory:
            self.current_trajectory = Trajectory()

        self._last_real_obs = obs.copy() if hasattr(obs, "copy") else np.array(obs)
        return result

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self.record_trajectory:
            s = self._last_real_obs
            s_next = obs
            self.current_trajectory.add_transition(
                state=s,
                action=action,
                reward=reward,
                next_state=s_next
            )
        self._last_real_obs = obs

        if self.record_trajectory and (done or truncated):
            self.trajectory_log.append(copy.deepcopy(self.current_trajectory))

        return obs, reward, done, truncated, info
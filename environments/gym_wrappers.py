from trajectory_graph import Transition
import gymnasium as gym


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
        self.current_trajectory = []     # Current episode
        self._last_real_obs = None

    def start_recording(self):
        self.record_trajectory = True
        self.current_trajectory = []

    def stop_recording(self):
        self.record_trajectory = False

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.record_trajectory:
            self.current_trajectory = []
        self._last_real_obs = obs.copy()
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.record_trajectory:
            self.current_trajectory.append(
                Transition(
                    state=self._last_real_obs.copy(),
                    action=action,
                    reward=reward,
                    next_state=obs.copy()
                )
            )
        self._last_real_obs = obs.copy()

        if self.record_trajectory and (done or truncated):
            self.trajectory_log.append(self.current_trajectory.copy())

        return obs, reward, done, truncated, info
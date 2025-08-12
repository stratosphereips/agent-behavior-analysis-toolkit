import numpy as np
from .gym_discretization_wrapper import DiscretizationWrapper
import gymnasium as gym

class DiscreteCartPoleWrapper(DiscretizationWrapper):
    """
    Wrapper for Gym Cartpole enviroment which converts the continous observation space to discrete state.
    """
    def __init__(self, env, bins=8):
        super().__init__(env, [
            np.linspace(-2.4, 2.4, num=bins + 1)[1:-1], # cart position
            np.linspace(-3, 3, num=bins + 1)[1:-1],     # cart velocity
            np.linspace(-0.2, 0.2, num=bins + 1)[1:-1], # pole angle
            np.linspace(-2, 2, num=bins + 1)[1:-1],     # pole angle velocity
        ])

class DiscreteMountainCarWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=None, tiles=None):
        if bins is None:
            bins = 24 if tiles is None or tiles <= 1 else 12 if tiles <= 3 else 8
        super().__init__(env, [
            np.linspace(-1.2, 0.6, num=bins + 1)[1:-1],   # car position
            np.linspace(-0.07, 0.07, num=bins + 1)[1:-1], # car velocity
        ], tiles)

class DiscreteLunarLanderWrapper(DiscretizationWrapper):
    def __init__(self, env):
        super().__init__(env, [
            np.linspace(-.4,   .4, num=5 + 1)[1:-1],   # x
            np.linspace(-.075,1.35,num=6 + 1)[1:-1],   # y
            np.linspace(-.5,   .5, num=5 + 1)[1:-1],   # vel x
            np.linspace(-.8,   .8, num=7 + 1)[1:-1],   # vel y
            np.linspace(-.2,   .2, num=3 + 1)[1:-1],   # rot
            np.linspace(-.2,   .2, num=5 + 1)[1:-1],   # ang vel
            [.5], #lc
            [.5], #rc
        ])

class DiscreteTaxiWrapper(DiscretizationWrapper):
    def __init__(self, env):
        super().__init__(env, [
            np.linspace(0, 4, num=5 + 1)[1:-1],   # taxi row
            np.linspace(0, 4, num=5 + 1)[1:-1],   # taxi column
            np.linspace(0, 4, num=5 + 1)[1:-1],   # passenger location
        ])

class DiscreteBlackJackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.player_bins = 18  # player_sum 4-21
        self.dealer_bins = 10  # dealer_card 1-10
        self.ace_bins = 2      # usable_ace False/True
        self.observation_space = gym.spaces.Discrete(self.player_bins * self.dealer_bins * self.ace_bins)
        self.action_space = self.env.action_space  # Actions remain unchanged
    
    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs

        player_bin = player_sum - 4        # 0..17
        dealer_bin = dealer_card - 1       # 0..9
        ace_bin = int(usable_ace)          # 0 or 1

        state_id = player_bin * self.dealer_bins * self.ace_bins + dealer_bin * self.ace_bins + ace_bin
        print(obs, state_id)
        return state_id
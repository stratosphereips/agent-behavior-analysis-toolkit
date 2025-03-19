import numpy as np
from .gym_discretization_wrapper import DiscretizationWrapper

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
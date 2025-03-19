import numpy as np
def get_bins(observation_space_high, observation_space_low, num_bins=10, clip_high=10, clip_low=-10)->list:
    """
    Crates bins in the specified range.
    Cliping is applied if `clip_low` and `clip_high`are provided.
    """
    bins = []
    for low,high in zip(observation_space_low, observation_space_high):
        # clip the values if required
        if clip_low:
            low = max(low, clip_low)
        if clip_high:
            high = min(high, clip_high)
        bins.append(np.linspace(low, high, num_bins))
    return bins

def discretize_observation(observation, bins)->tuple:
    """
    Transforms array of continuous observations into discrere values (in given bins)
    """
    discretized = []
    for i, value in enumerate(observation):
        discretized.append(np.digitize(value, bins[i]) - 1)
    return tuple(discretized)
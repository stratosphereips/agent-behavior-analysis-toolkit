from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Any, Iterable, Hashable
import copy
import numpy as np

# Transition class to represent a single transition in the trajectory
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

@dataclass
class Trajectory:
    """
    Class to represent a trajectory (sequence of transitions) in the environment.
    """
    transitions: list[Transition] = field(default_factory=list)


    def add_transition(self, state, action, reward, next_state) -> None:
        """
        Add a transition to the trajectory.
        """
        self.transitions.append(Transition(state, action, reward, next_state))
    
    def __len__(self)->int:
        """
        Return the number of transitions in the trajectory.
        """
        return len(self.transitions)
    
    def __iter__(self):
        """
        Allow iteration over the transitions in the trajectory.
        """
        return iter(self.transitions)
    
    def __getitem__(self, index)-> Transition:
        """
        Get a transition by index.
        """
        return self.transitions[index]
    
    def total_reward(self)-> float:
        """
        Calculate the total reward of the trajectory.
        """
        return sum(transition.reward for transition in self.transitions)
    
    @property
    def actions(self)-> list:
        """
        Get a list of actions in the trajectory.
        """
        return [transition.action for transition in self.transitions]
    
    @property
    def states(self)-> list:
        """
        Get a list of states in the trajectory. Add the next state of the last transition to the list.
        """
        return [transition.state for transition in self.transitions] + [self.transitions[-1].next_state] if self.transitions else []
    
    @property
    def rewards(self)-> list:
        """
        Get a list of rewards in the trajectory.
        """
        return [transition.reward for transition in self.transitions]
    
    def __str__(self) -> str:
        """
        String representation of the trajectory.
        """
        return " -> ".join(f"{t.state} -{t.action}-> {t.next_state} (r={t.reward})" for t in self.transitions)
    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False
        if len(self) != len(other):
            return False
        for (t1,t2) in zip(self.transitions, other.transitions):
            if t1 != t2:
                return False
        return True
    def __hash__(self):
        return hash(str(self))
        
class Policy():
    """
    Abstract base class for policies.
    Policies define how an agent behaves in the environment.
    """
    def select_action(self, observation: Any) -> Any:
        """
        Select an action based on the current observation.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def initial_state(self) -> Any:
        """
        Return the initial state of the policy.
        Default is None
        """
        return None
    def get_action_probability(self, state: Any, action: Any) -> float:
        """
        Get the probability of taking a specific action in a given state.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class EmpiricalPolicy(Policy):
    """
    Class representing an empirical policy based on a set of trajectories.
    The policy is defined by the most frequent action taken in each state.
    """
    def __init__(self, trajectories: Iterable[Trajectory]):
        # Store the trajectories and build the policy from them
        self.trajectories = []
        self._state_action_map = {}
        self._edge_count = {}
        self._edge_reward = {}
        self.update_policy(trajectories)
    @property
    def num_states(self)->int:
        """
        Return number of unique states in the observed trajectories
        """
        states = set([state for state in self._state_action_map.keys()])
        return len(states)
    
    @property
    def num_actions(self)->int:
        actions = set()
        for action_dict in self._state_action_map.values():
            actions.update(action_dict.keys())
        return len(actions)
    @property
    def num_trajectories(self) -> int:
        """
        Return the number of trajectories in the policy.
        """
        return len(self.trajectories)
    def _convert_to_hashable(self, value:Any)->Hashable:
        """
        Convert a value to a hashable type.
        """
        if isinstance(value, np.ndarray):
            if np.ndim(value) == 0:
                return int(value)
            else:
                return tuple(value)
        return value
        
    def update_policy(self, new_trajectories: Iterable[Trajectory]):
        for trajectory in new_trajectories:  
            self.add_trajectory(trajectory)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add a new trajectory and update the policy.
        """
        if not isinstance(trajectory, Trajectory):
            raise ValueError("Expected a Trajectory instance.")
        self.trajectories.append(trajectory)
        for transition in trajectory:
            self._add_transition(transition)
    
    def _add_transition(self, transition: Transition) -> None:
        """
        Add a single transition to the policy.
        """
       
        state = self._convert_to_hashable(transition.state)
        next_state = self._convert_to_hashable(transition.next_state)
        action = self._convert_to_hashable(transition.action)

        if state not in self._state_action_map:
            self._state_action_map[state] = {}
        if action not in self._state_action_map[state]:
            self._state_action_map[state][action] = 0
        self._state_action_map[state][action] += 1
        self._edge_count[(state, action, next_state)] = self._edge_count.get((state, action, next_state), 0) + 1
        self._edge_reward[(state, action, next_state)] = self._edge_reward.get((state, action, next_state), 0) + transition.reward

    def get_action_probability(self, state: Any, action: Any, alpha=0.1) -> float:
        """
        Get the Laplace-smoothed probability of taking a specific action in a given state.
        If the state is unseen, assume uniform probability over the action space.
        
        Parameters:
            state: the state
            action: the action
            alpha: smoothing constant (default = 1.0)

        Returns:
            Smoothed probability π(a | s)
        """
        state = self._convert_to_hashable(state)
        action = self._convert_to_hashable(action)
        if state not in self._state_action_map:
            # Unseen state: return uniform probability
            return 1.0 / self.num_actions

        action_counts = self._state_action_map[state]
        total_count = sum(action_counts.get(a, 0) for a in range(self.num_actions))

        # Apply Laplace smoothing
        numerator = action_counts.get(action, 0) + alpha
        denominator = total_count + alpha * self.num_actions

        return numerator / denominator
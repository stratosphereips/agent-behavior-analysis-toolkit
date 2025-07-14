from collections import namedtuple
from dataclasses import dataclass, field

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
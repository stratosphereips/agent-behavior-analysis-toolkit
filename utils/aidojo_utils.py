from AIDojoCoordinator.game_components import GameState, Action,ActionType
from AIDojoCoordinator.utils.utils import state_as_ordered_string
from trajectory import Trajectory
import json
from typing import Any, Dict, Iterable

def aidojo_rebuild_trajectory(states:Iterable[Dict[str, Any]], actions:Iterable[Dict[str, Any]], rewards:Iterable[float])->Trajectory:
    """
    Rebuild a Trajectory object from its components.
    """
    states = [state_as_ordered_string(GameState.from_dict(s)) if isinstance(s, dict) else s for s in states]
    actions = [Action.from_dict(a) if isinstance(a, dict) else a for a in actions]
    traj = Trajectory()
    for s, a, r, s_next in zip(states, actions, rewards, states[1:]):
        traj.add_transition(s, a, r, s_next)
    return traj


def aidojo_state_str_from_dict(state: Dict[str, Any]) -> str:
    """
    Encode a GameState into a string representation.
    """
    return state_as_ordered_string(GameState.from_dict(state))

def aidojo_action_from_dict(action: Dict[str, Any]) -> Action:
    """
    Decode an Action from its dictionary representation.
    """
    return Action.from_dict(action)

def aidojo_action_type_from_dict(action: Dict[str, Any]) -> ActionType:
    """
    Decode action from its dictionary representation and keep only the type information (discard action parameters).
    """
    return Action.from_dict(action).type
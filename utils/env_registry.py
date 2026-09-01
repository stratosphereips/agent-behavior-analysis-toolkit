"""
Registry of per-environment state/action decoders used when loading trajectories.

Trajectory files store states and actions however the environment that recorded them
produced them: plain values for gym-style envs, dicts for netsecgame. Everything else
about loading (walking directories, parsing JSON/JSONL, building Trajectory/EmpiricalPolicy
objects) is already environment-agnostic (utils.trajectory_utils, utils.data_utils).
So the only thing a new environment needs to plug in is a pair of decode functions.

To add a new environment:

    from utils.env_registry import EnvCodec, register_env

    def my_env_decode_state(state: dict): ...
    def my_env_decode_action(action: dict): ...

    register_env("my_env", EnvCodec(
        state_encoder=my_env_decode_state,
        action_encoder=my_env_decode_action,
    ))

Then pass env="my_env" wherever trajectories are loaded (e.g.
utils.trajectory_utils.load_trajectories(path, env="my_env")) instead of wiring the
decode callables by hand at each call site.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict

from utils.aidojo_utils import aidojo_state_str_from_dict, aidojo_action_type_from_dict


def _identity(x: Any) -> Any:
    return x


@dataclass(frozen=True)
class EnvCodec:
    """
    Decode functions applied to raw JSON values while loading a trajectory file.
    Names match the `state_encoder`/`action_encoder` parameters of
    utils.trajectory_utils.load_trajectories_from_json/_jsonl, which they are passed to.
    """
    state_encoder: Callable[[Any], Any] = _identity
    action_encoder: Callable[[Any], Any] = _identity


_REGISTRY: Dict[str, EnvCodec] = {}


def register_env(name: str, codec: EnvCodec, overwrite: bool = False) -> None:
    """Register a new environment's decode functions under `name`."""
    if not overwrite and name in _REGISTRY:
        raise ValueError(f"Environment '{name}' is already registered (pass overwrite=True to replace it).")
    _REGISTRY[name] = codec


def get_env_codec(name: str) -> EnvCodec:
    """Look up a registered environment's decode functions by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown environment '{name}'. Registered environments: {sorted(_REGISTRY)}. "
            "Use register_env() to add a new one."
        ) from None


# Gym-style envs (taxi, frozenlake, cartpole, ...): states/actions need no decoding.
register_env("numpy", EnvCodec())
register_env("default", EnvCodec())

# netsecgame/aidojo: states are decoded to the ordered-string GameState representation,
# actions are decoded down to just their ActionType (discarding parameters).
_netsecgame_codec = EnvCodec(
    state_encoder=aidojo_state_str_from_dict,
    action_encoder=aidojo_action_type_from_dict,
)
register_env("netsecgame", _netsecgame_codec)
register_env("aidojo", _netsecgame_codec)

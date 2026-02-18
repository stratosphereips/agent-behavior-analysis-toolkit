
import json
import os
import numpy as np
import jsonlines
from typing import Callable, Any, Optional
import re
from trajectory import Trajectory, Transition

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class TrajectoryRecorder:
    """
    Records trajectories and saves them to a JSONL file.
    Supports custom encoders for states and actions.
    """
    def __init__(
        self, 
        log_path: str, 
        state_encoder: Optional[Callable[[Any], Any]] = None, 
        action_encoder: Optional[Callable[[Any], Any]] = None,
        auto_flush: bool = True
    ):
        """
        Args:
            log_path (str): Path to the output JSONL file.
            state_encoder (Callable, optional): Function to encode state before storing.
            action_encoder (Callable, optional): Function to encode action before storing.
            auto_flush (bool): If True, writes to file immediately on end_trajectory.
        """
        if log_path:
            # Enforce 4-digit zero padding for cp_X.jsonl files
            dirname, filename = os.path.split(log_path)
            match = re.match(r"^cp_(\d+)\.jsonl$", filename)
            if match:
                num = int(match.group(1))
                new_filename = f"cp_{num:05d}.jsonl"
                log_path = os.path.join(dirname, new_filename)
        
        self.log_path = log_path
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.auto_flush = auto_flush
        
        # Ensure directory exists
        if log_path:
             os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        
        self.current_trajectory: Optional[Trajectory] = None
        self.current_metadata: dict = {}

    def start_trajectory(self, metadata: dict = None) -> None:
        """
        Starts a new trajectory recording.
        Args:
            metadata (dict, optional): Metadata associated with this trajectory (e.g., config, ids).
        """
        self.current_trajectory = Trajectory()
        self.current_metadata = metadata if metadata else {}

    def add_transition(self, state: Any, action: Any, reward: float, next_state: Any) -> None:
        """
        Adds a transition to the current trajectory.
        Encodes state and action if encoders are provided.
        """
        if self.current_trajectory is None:
            raise RuntimeError("Called add_transition before start_trajectory")

        encoded_state = self.state_encoder(state) if self.state_encoder else state
        encoded_action = self.action_encoder(action) if self.action_encoder else action
        encoded_next_state = self.state_encoder(next_state) if self.state_encoder else next_state

        self.current_trajectory.add_transition(encoded_state, encoded_action, reward, encoded_next_state)

    def end_trajectory(self, save: bool = True) -> None:
        """
        Ends the current trajectory.
        Args:
            save (bool): If True, saves the trajectory to file.
        """
        if self.current_trajectory is None:
            return Warning("Called end_trajectory before start_trajectory")

        if save and self.auto_flush:
            self._save_trajectory()
        
        # Reset
        self.current_trajectory = None
        self.current_metadata = {}

    def _save_trajectory(self) -> None:
        """Appends the current trajectory to the JSONL file."""
        if not self.current_trajectory:
            return

        json_data = self.current_trajectory.to_json(metadata=self.current_metadata)
        
        output_obj = {
            "trajectory": {
                "states": json_data["states"],
                "actions": json_data["actions"],
                "rewards": json_data["rewards"]
            },
            "metadata": self.current_metadata
        }
        
        # Define a custom dumper that uses our CustomJSONEncoder
        def numpy_dumps(obj):
            return json.dumps(obj, cls=CustomJSONEncoder)
        
        # Append to file using jsonlines
        # mode='a' for appending
        with jsonlines.open(self.log_path, mode='a', dumps=numpy_dumps) as writer:
            writer.write(output_obj)

import pytest
import os
import json
import sys
from utils.recorder import TrajectoryRecorder
from utils.trajectory_utils import load_trajectories_from_jsonl

def test_recorder_basic(tmp_path):
    log_file = tmp_path / "test_log.jsonl"
    recorder = TrajectoryRecorder(str(log_file))
    
    recorder.start_trajectory(metadata={"id": 1})
    recorder.add_transition(0, "A", 1.0, 1)
    recorder.add_transition(1, "B", 0.0, 2)
    recorder.end_trajectory()
    
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        
    assert data["metadata"]["id"] == 1
    assert data["trajectory"]["actions"] == ["A", "B"]
    assert data["trajectory"]["rewards"] == [1.0, 0.0]
    assert data["trajectory"]["states"] == [0, 1, 2]

def test_recorder_encoders(tmp_path):
    log_file = tmp_path / "encoded_log.jsonl"
    
    def state_enc(s):
        return f"s_{s}"
        
    def action_enc(a):
        return a * 2
        
    recorder = TrajectoryRecorder(str(log_file), state_encoder=state_enc, action_encoder=action_enc)
    
    recorder.start_trajectory()
    recorder.add_transition(0, 1, 10.0, 1) # Action 1 -> 2
    recorder.end_trajectory()
    
    with open(log_file, 'r') as f:
        data = json.loads(f.readline())
        
    assert data["trajectory"]["states"] == ["s_0", "s_1"]
    assert data["trajectory"]["actions"] == [2]

def test_recorder_load_compatibility(tmp_path):
    """Verify that recorded logs can be loaded back using trajectory_utils."""
    log_file = tmp_path / "compat_log.jsonl"
    recorder = TrajectoryRecorder(str(log_file))
    
    recorder.start_trajectory({"run": "test"})
    recorder.add_transition("start", "go", 5.0, "end")
    recorder.end_trajectory()
    
    # Load back using the utility
    trajectories, metadata = load_trajectories_from_jsonl(str(log_file), load_metadata=True)
    
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert len(traj) == 1
    assert traj[0].state == "start"
    assert traj[0].action == "go"
    # Metadata loading in utility merges metadata from all lines/single dict. 
    # Current load_trajectories_from_jsonl implementation updates a single metadata dict.
    assert metadata["run"] == "test"

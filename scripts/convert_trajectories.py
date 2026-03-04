import os
import json
import copy
from typing import List, Any
import gzip
from pathlib import Path
from netsecgame import Action
# Adjust imports to match actual module path for Trajectory
# Assuming this script is run from the agent_trajectory_analysis directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from trajectory import Trajectory, Transition

def convert_trajectory_file(file_path: str) -> tuple[Trajectory, dict]:
    """
    Parses a non-standard JSONL trajectory file and returns a Trajectory object and metadata.
    """
    trajectory = Trajectory()
    
    # We might need to keep track of the previous state to form transitions
    prev_state = None
    prev_action = None
    prev_reward = 0.0
    
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'
    
    metadata = {
        "source_file": Path(file_path).name,
        "steps": []
    }
    
    with open_func(file_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if data.get("type") == "meta":
                # Metadata line, can be used to set trajectory metadata if needed
                for k, v in data.items():
                    if k != "type":
                        metadata[k] = v
                continue
            elif data.get("type") == "step" or "state_vec" in data:
                # Step data
                state = data.get("state_vec")
                if state is not None:
                    state = tuple(state) # Convert to a hashable type if needed by empirical policy later
                
                # Extract action
                chosen = data.get("chosen", {})
                action_type = chosen.get("action_type")
                params = chosen.get("params", {})
                
                # Form a sensible hashable action representation
                #action_str = f"{action_type}_{json.dumps(params, sort_keys=True)}" if action_type else None
                action_dict = {"action_type": action_type, "parameters": params}
                action = Action.from_dict(action_dict)
                reward = data.get("reward", 0.0)
                done = data.get("done", False)
                
                step_meta = {}
                for k, v in data.items():
                    if k not in ["type", "state_vec", "chosen", "reward"]:
                        step_meta[k] = v
                metadata["steps"].append(step_meta)
                
                if prev_state is not None:
                    trajectory.add_transition(
                        state=prev_state,
                        action=prev_action.as_dict,
                        reward=prev_reward, # We need to store prev_reward too
                        next_state=state
                    )
                
                prev_state = state
                prev_action = action
                prev_reward = reward
                
                # Handle terminal state
                if done:
                    pass
    
    return trajectory, metadata

def convert_all(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "eval"]:
        split_dir = input_path / split
        if not split_dir.exists():
            print(f"Directory not found: {split_dir}")
            continue
            
        print(f"Processing split: {split}")
        out_file = output_path / f"{split}.jsonl"
        print(f"Saving merged {split} trajectories to {out_file}...")
        
        with open(out_file, 'w') as f_out:
            for file in split_dir.glob("*.jsonl*"): # Handles .jsonl and .jsonl.gz
                print(f"Converting {file}...")
                traj, meta = convert_trajectory_file(str(file))
                f_out.write(json.dumps({"trajectory": traj.to_json(metadata=meta), "metadata": meta}) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert trajectory files.")
    parser.add_argument("--input_dir", type=str, default="/home/ondra/trajectories/216_252_traj", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="/home/ondra/trajectories/216_252_traj_converted", help="Output directory")
    args = parser.parse_args()
    
    convert_all(args.input_dir, args.output_dir)
    print("Conversion complete.")

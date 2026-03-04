import os
import json
import gzip
from pathlib import Path
from typing import List, Any, Tuple
import sys

from netsecgame import Action

sys.path.append(str(Path(__file__).parent.parent))
from trajectory import Trajectory, Transition

def convert_trajectory_file(file_path: str) -> Tuple[Trajectory, dict]:
    """
    Parses a MAML non-standard JSONL trajectory file and returns a Trajectory object and metadata.
    """
    trajectory = Trajectory()
    
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
            
            if data.get("record_type") == "trajectory_header":
                for k, v in data.items():
                    if k != "record_type":
                        metadata[k] = v
                continue
            elif data.get("record_type") == "step" or "nn_input" in data:
                state = data.get("nn_input")
                if state is not None:
                    state = tuple(state)
                
                chosen = data.get("selected_action", {})
                action_type = chosen.get("action_type")
                params = chosen.get("params", {})
                
                # In MAML JSONL, hosts and networks are passed as strings, but Action.from_dict expects mappings
                for k in list(params.keys()):
                    if isinstance(params[k], str):
                        if k.endswith("_host") or k == "host" or k == "ip":
                            params[k] = {"ip": params[k]}
                        elif k.endswith("_network") or k == "network" or k == "subnet":
                            if "/" in params[k]:
                                ip, mask = params[k].split("/")
                                params[k] = {"ip": ip, "mask": int(mask)}
                            else:
                                params[k] = {"ip": params[k], "mask": 24}
                        elif k.endswith("_service") or k == "service":
                            params[k] = {"name": params[k]}
                        elif k.endswith("_data") or k == "data":
                            if ":" in params[k]:
                                owner, data_id = params[k].split(":", 1)
                                params[k] = {"owner": owner, "id": data_id}
                            else:
                                params[k] = {"owner": "Unknown", "id": params[k]}
                        
                action_dict = {"action_type": action_type, "parameters": params}
                action = Action.from_dict(action_dict)
                reward = data.get("reward_after", 0.0)
                done = data.get("done_next", False)
                
                step_meta = {}
                for k, v in data.items():
                    if k not in ["record_type", "nn_input", "selected_action", "reward_after"]:
                        step_meta[k] = v
                metadata["steps"].append(step_meta)
                
                if prev_state is not None:
                    trajectory.add_transition(
                        state=prev_state,
                        action=prev_action.as_dict,
                        reward=prev_reward,
                        next_state=state
                    )
                
                prev_state = state
                prev_action = action
                prev_reward = reward
                
                if done:
                    # Flush the final transition if done
                    trajectory.add_transition(
                        state=prev_state,
                        action=prev_action.as_dict,
                        reward=prev_reward,
                        next_state=state # Or None if preferred, using state as fallback
                    )
                    prev_state = None
                    prev_action = None
                    prev_reward = 0.0
                    
        # Flush the last transition if the file ended without done=True
        if prev_state is not None:
            trajectory.add_transition(
                state=prev_state,
                action=prev_action.as_dict,
                reward=prev_reward,
                next_state=prev_state
            )

    return trajectory, metadata

def convert_all(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if there are splits or just jsonl files directly
    jsonl_files = list(input_path.glob("*.jsonl*"))
    has_splits = False
    
    if jsonl_files:
        print(f"Found {len(jsonl_files)} files in {input_path}")
        out_file = output_path / "all_trajectories.jsonl"
        print(f"Saving combined trajectories to {out_file}...")
        with open(out_file, 'w') as f_out:
            for file in jsonl_files:
                # print(f"Converting {file}...")
                traj, meta = convert_trajectory_file(str(file))
                f_out.write(json.dumps({"trajectory": traj.to_json(metadata=meta), "metadata": meta}) + '\n')
    else:
        for split in ["train", "eval", "seen_raw"]:
            split_dir = input_path / split
            if not split_dir.exists():
                continue
            has_splits = True
            print(f"Processing split: {split}")
            out_file = output_path / f"{split}.jsonl"
            print(f"Saving merged {split} trajectories to {out_file}...")
            
            with open(out_file, 'w') as f_out:
                for file in split_dir.glob("*.jsonl*"):
                    # print(f"Converting {file}...")
                    traj, meta = convert_trajectory_file(str(file))
                    f_out.write(json.dumps({"trajectory": traj.to_json(metadata=meta), "metadata": meta}) + '\n')
                    
        if not has_splits:
            print(f"No jsonl files or split directories found in {input_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert MAML trajectory files.")
    parser.add_argument("--input_dir", type=str, default="/datafast/ondra/trajectories/aidojo-utep/jihoon_maml/seen_raw", help="Input directory containing raw jsonl trajectories.")
    parser.add_argument("--output_dir", type=str, default="/datafast/ondra/trajectories/aidojo-utep/jihoon_maml/seen_converted", help="Output directory for converted trajectories.")
    args = parser.parse_args()
    
    convert_all(args.input_dir, args.output_dir)
    print("Conversion complete.")

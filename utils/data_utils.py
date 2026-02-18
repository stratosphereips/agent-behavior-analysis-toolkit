import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from trajectory import EmpiricalPolicy, Trajectory
from utils.trajectory_utils import load_trajectories_from_json
from utils.aidojo_utils import aidojo_action_type_from_dict, aidojo_state_str_from_dict
import random


def _process_file(path, max_trajectories, action_encoder, state_encoder, test_split:float|None=None)->tuple[EmpiricalPolicy, list[Trajectory], EmpiricalPolicy|None, EmpiricalPolicy|None]:
    """
    Worker function to process a single trajectory file.

    Args:
        path (str): Path to the JSON file containing trajectories.
        max_trajectories (int): Maximum number of trajectories to load.
        action_encoder (Callable, optional): Function to encode actions.
        state_encoder (Callable, optional): Function to encode states.
    Returns:
        tuple[EmpiricalPolicy, list[Trajectory], EmpiricalPolicy|None, EmpiricalPolicy|None]: The constructed empirical policy and list of loaded trajectories.
    """
    trajectories, metadata = load_trajectories_from_json(
        path, 
        max_trajectories=max_trajectories, 
        load_metadata=True,
        action_encoder=action_encoder,
        state_encoder=state_encoder
    )
    trajectories = trajectories[:max_trajectories]
    print(f"[Trajectory processing & EP build] {path}")
    full_policy = EmpiricalPolicy(trajectories, metadata=metadata)
    if test_split is not None:
        print(f"[Trajectory processing & EP build] Splitting into train/test with split {test_split}")
        train_size = int(len(trajectories) * (1-test_split))
        test_size = int(len(trajectories) * test_split)
        print(f"[Trajectory processing & EP build] Train size: {train_size}, Test size: {test_size}")
        shuffled_trajectories = trajectories.copy()
        print(len(shuffled_trajectories))
        random.shuffle(shuffled_trajectories)   
        print("Shuffled trajectories")
        test_trajectories = shuffled_trajectories[-test_size:]
        train_trajectories = shuffled_trajectories[:-test_size]
        train_policy = EmpiricalPolicy(train_trajectories, metadata=metadata)
        test_policy = EmpiricalPolicy(test_trajectories, metadata=metadata)
        print("Split policies done")
        return full_policy, trajectories, train_policy, test_policy
    return full_policy, trajectories, None, None

def _insert_nested(root_dict, keys:tuple, value):
    """
    Helper to insert value into a nested dictionary creating intermediate dicts as needed.

    Args:
        root_dict (dict): The root dictionary to insert into.
        keys (tuple): Tuple of keys to navigate to the target location.
        value: The value to insert.
    """
    current = root_dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def get_nested_paths(directory, sort_keys=True)->dict[str, dict[str, str]]:
    """
    Recursively scans a directory and returns a nested dictionary of file paths.
    Keys are directory/file names, values are either nested dictionaries or absolute file paths.
    Only includes .json and .jsonl files.
    """
    results = {}
    for entry in os.scandir(directory):
        if entry.is_dir():
            sub_results = get_nested_paths(entry.path, sort_keys=sort_keys)
            if sub_results: # Only add if it contains relevant files
                results[entry.name] = sub_results
        elif entry.is_file() and (entry.name.endswith('.json') or entry.name.endswith('.jsonl')):
            key_name = os.path.splitext(entry.name)[0]
            results[key_name] = entry.path
            
    if sort_keys:
        return dict(sorted(results.items()))
    return results

def load_policies_from_directory(directory, max_trajectories=None, action_encoder=None, state_encoder=None, sort_keys=True, test_split:float|None=None)->dict[str, dict[str, tuple[EmpiricalPolicy, list[Trajectory], EmpiricalPolicy|None, EmpiricalPolicy|None]]]:
    """
    Loads empirical policies from a directory structure.
    Returns:
        dict: Nested dictionary matching the directory structure.
              Values are (EmpiricalPolicy, list[Trajectory]).
    """
    paths = get_nested_paths(directory, sort_keys=sort_keys)
    return load_policies_from_paths(paths, max_trajectories, action_encoder, state_encoder, test_split=test_split)

def load_policies_from_paths(nested_paths, max_trajectories, action_encoder=None, state_encoder=None, test_split:float|None=None)->dict[str, dict[str, tuple[EmpiricalPolicy, list[Trajectory], EmpiricalPolicy|None, EmpiricalPolicy|None]]]:
    """
    Loads policies given a nested dictionary of paths (output of get_nested_paths).

    Args:
        nested_paths (dict): Nested dictionary of file paths.
        max_trajectories (int): Maximum number of trajectories to load.
        action_encoder (ActionEncoder, optional): Action encoder to use. Defaults to None.
        state_encoder (StateEncoder, optional): State encoder to use. Defaults to None.
    Returns:
        dict: Nested dictionary matching the directory structure.
              Values are (EmpiricalPolicy, list[Trajectory]).
    """
    # 1. Flatten the nested structure to identify all tasks
    tasks = [] # list of (key_path_tuple, file_path)
    
    def gather_tasks(d, current_key_path):
        for key, value in d.items():
            new_path = current_key_path + (key,)
            if isinstance(value, dict):
                gather_tasks(value, new_path)
            else:
                tasks.append((new_path, value))
    
    gather_tasks(nested_paths, ())
    # 2. Parallel Execution
    results_flat = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_process_file, path, max_trajectories, action_encoder, state_encoder, test_split): key_path
            for (key_path, path) in tasks
        }
        
        for f in as_completed(futures):
            key_path = futures[f]
            try:
                policy, trajectories, train_policy, test_policy = f.result()
                results_flat[key_path] = (policy, trajectories, train_policy, test_policy)
            except Exception as e:
                print(f"Error loading {key_path}: {e}")

    # 3. Reconstruct Nested Structure
    # We copy the structure of nested_paths but replace leaves
    def reconstruct(d, current_key_path):
        new_d = {}
        for key, value in d.items():
            new_path = current_key_path + (key,)
            if isinstance(value, dict):
                new_d[key] = reconstruct(value, new_path)
            else:
                if new_path in results_flat:
                    new_d[key] = results_flat[new_path]
                else:
                    new_d[key] = None # Failed or missing
        return new_d

    return reconstruct(nested_paths, ())

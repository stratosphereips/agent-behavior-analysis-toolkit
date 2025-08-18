from utils.trajectory_utils import load_trajectories_from_json
from trajectory import EmpiricalPolicy
import os


class TrajectoryReplay:
    """
    Class for loading and processing of recorded trajectories
    """

    def __init__(self, trajectory_dir, **kwargs):
        self.trajectories = []
        self.json_files = sorted([os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir) if f.endswith(".json")])
        print(f"Found {len(self.json_files)} JSON files in {trajectory_dir}")
        self._previous_policy = None
        self._wandb_run = kwargs.get("wandb_run", None)

    def process_trajectories(self):
        for checkpoint_id, json_file in enumerate(self.json_files):
            trajectories = load_trajectories_from_json(json_file)
            print(f"Loaded {len(trajectories)} trajectories from {json_file}: - checkpoint {checkpoint_id}")
            log_data = {
                "checkpoint_id": checkpoint_id,
                "num_trajectories": len(trajectories),
            }
            # Do something with the loaded trajectories
            self.trajectories.extend(trajectories)
            empirical_policy = EmpiricalPolicy(self.trajectories)


if __name__ == "__main__":
    trajectory_replay = TrajectoryReplay("trajectories/")
    trajectory_replay.process_trajectories()
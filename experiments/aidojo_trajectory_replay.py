import sys
from .trajectory_replay import TrajectoryReplay
from trajectory import Trajectory
from trajectory import Transition


class AIDojoTrajectoryReplay(TrajectoryReplay):
    def __init__(self, trajectory_dir, **kwargs):
        super().__init__(trajectory_dir=trajectory_dir, **kwargs)
        self.state_to_id = {}
        self.action_to_id = {}

    def get_state_id(self, state):
        if state not in self.state_to_id:
            self.state_to_id[state] = len(self.state_to_id)
        return self.state_to_id[state]

    def get_action_id(self, action):
        if action not in self.action_to_id:
            self.action_to_id[action] = len(self.action_to_id)
        return self.action_to_id[action]

    def remap_trajectories(self, trajectories:dict)-> dict:
        """
        Remap states and actions in trajectories to integer IDs.
        Args:
            trajectories (dict): Dictionary of trajectories per checkpoint.
        Returns:
            dict: Remapped trajectories with integer state and action IDs.
        """
        print("[remap_trajectories] Remapping states and actions to integer IDs")
        new_trajectories = {}
        for checkpoint_id, trajectories in trajectories.items():
            new_trajectories[checkpoint_id] = []
            for traj in trajectories:
                new_transitions = []
                for transition in traj.transitions:
                    new_transition = Transition(
                        state=self.get_state_id(transition.state),
                        action=self.get_action_id(transition.action),
                        reward=transition.reward,
                        next_state=self.state_to_id.get(transition.next_state, -1),
                    )
                    new_transitions.append(new_transition)
                new_trajectories[checkpoint_id].append(Trajectory(transitions=new_transitions))
        print(f"[remap_trajectories] Remapped {len(self.state_to_id)} unique states and {len(self.action_to_id)} unique actions")
        return new_trajectories

if __name__ == "__main__":
    trajectory_replay = AIDojoTrajectoryReplay(sys.argv[1],
    wandb_project="agent-trajectory-analysis",
    wandb_entity="ondrej-lukas-czech-technical-university-in-prague"
    )
    trajectory_replay.process_trajectories()

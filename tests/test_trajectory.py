
import pytest
import numpy as np
from dataclasses import dataclass, is_dataclass
from trajectory import Trajectory, Transition, EmpiricalPolicy

def test_transition_structure():
    """Test that Transition is a dataclass and has correct fields."""
    # This might fail before refactor if it's still a namedtuple, 
    # but we are designing tests for the target state.
    # To make it runnable before refactor, we can check if it's namedtuple or dataclass
    
    t = Transition(state=1, action=2, reward=3.0, next_state=4)
    
    assert t.state == 1
    assert t.action == 2
    assert t.reward == 3.0
    assert t.next_state == 4
    
    # Check equality
    t2 = Transition(state=1, action=2, reward=3.0, next_state=4)
    assert t == t2
    
    # Check hashing if it's hashable (namedtuple is, frozen dataclass is)
    assert hash(t) == hash(t2)

def test_trajectory_init():
    """Test Trajectory initialization."""
    traj = Trajectory()
    assert len(traj) == 0
    assert traj.total_reward() == 0.0
    assert traj.states == []
    assert traj.actions == []
    assert traj.rewards == []

def test_trajectory_add_transition():
    """Test adding transitions to trajectory."""
    traj = Trajectory()
    traj.add_transition(state=0, action='a', reward=1.0, next_state=1)
    
    assert len(traj) == 1
    assert traj.total_reward() == 1.0
    assert traj.states == [0, 1]
    assert traj.actions == ['a']
    assert traj.rewards == [1.0]
    
    t = traj[0]
    assert t.state == 0
    assert t.action == 'a'
    assert t.reward == 1.0
    assert t.next_state == 1

def test_trajectory_json_serialization():
    """Test to_json and from_json."""
    traj = Trajectory()
    traj.add_transition(0, 'a', 1.0, 1)
    traj.add_transition(1, 'b', 0.0, 2)
    
    expected_meta = {"source": "test"}
    json_data = traj.to_json(metadata=expected_meta)
    
    assert json_data['states'] == [0, 1, 2]
    assert json_data['actions'] == ['a', 'b']
    assert json_data['rewards'] == [1.0, 0.0]
    assert json_data['metadata'] == expected_meta
    
    traj_loaded = Trajectory.from_json(json_data)
    assert traj == traj_loaded
    assert len(traj_loaded) == 2

def test_empirical_policy_basics():
    """Test EmpiricalPolicy stats and initialization."""
    t1 = Trajectory()
    t1.add_transition(0, 0, 1.0, 1)
    
    t2 = Trajectory()
    t2.add_transition(0, 1, 0.0, 2)
    
    policy = EmpiricalPolicy([t1, t2])
    
    # States: 0, 1, 2
    assert policy.num_states == 3
    # Actions: 0, 1
    assert policy.num_actions == 2
    assert policy.num_trajectories == 2
    assert policy.mean_return == 0.5
    
    assert policy.has_data(0)
    assert not policy.has_data(3)

def test_empirical_policy_probabilities():
    """Test probability calculations."""
    t1 = Trajectory()
    t1.add_transition(0, 'A', 1.0, 1) # 0 -> A
    t2 = Trajectory()
    t2.add_transition(0, 'A', 1.0, 1) # 0 -> A
    t3 = Trajectory()
    t3.add_transition(0, 'B', 1.0, 2) # 0 -> B
    
    policy = EmpiricalPolicy([t1, t2, t3])
    
    # State 0 has 2 'A's and 1 'B'. Total 3.
    # Laplace smoothing with alpha=0.1
    # num_actions = 2 ('A', 'B')
    # Prob(A|0) = (2 + 0.1) / (3 + 0.1 * 2) = 2.1 / 3.2 = 0.65625
    # Prob(B|0) = (1 + 0.1) / (3 + 0.1 * 2) = 1.1 / 3.2 = 0.34375
    
    prob_A = policy.get_action_probability(0, 'A', alpha=0.1)
    prob_B = policy.get_action_probability(0, 'B', alpha=0.1)
    
    np.testing.assert_allclose(prob_A, 2.1/3.2)
    np.testing.assert_allclose(prob_B, 1.1/3.2)

def test_empirical_policy_explicit_action_space():
    """Test using explicit action space to account for unseen actions."""
    t1 = Trajectory()
    t1.add_transition(0, 'A', 1.0, 1)
    
    # Action space includes 'C', which is never seen
    action_space = ['A', 'B', 'C']
    policy = EmpiricalPolicy([t1], action_space=action_space)
    
    assert policy.num_actions == 3
    
    # Prob calc for state 0 (1 observation of 'A')
    # Alpha = 0.1
    # Total count = 1
    # Denom = 1 + 0.1 * 3 = 1.3
    # Prob(A) = (1 + 0.1) / 1.3 = 1.1 / 1.3
    # Prob(B) = (0 + 0.1) / 1.3 = 0.1 / 1.3
    # Prob(C) = (0 + 0.1) / 1.3 = 0.1 / 1.3
    
    prob_C = policy.get_action_probability(0, 'C', alpha=0.1)
    np.testing.assert_allclose(prob_C, 0.1/1.3)

def test_empirical_policy_average_value():
    """Test average value of entering a state."""
    # Trajectory 1: 0 -(r=10)-> 1
    # Trajectory 2: 0 -(r=20)-> 1
    # Trajectory 3: 2 -(r=5)-> 1
    
    t1 = Trajectory()
    t1.add_transition(0, 'a', 10.0, 1)
    t2 = Trajectory()
    t2.add_transition(0, 'a', 20.0, 1)
    t3 = Trajectory()
    t3.add_transition(2, 'a', 5.0, 1)
    
    policy = EmpiricalPolicy([t1, t2, t3])
    
    # State 1 incoming rewards: 10, 20, 5. Average = 35 / 3
    assert policy.get_average_value(1) == pytest.approx(35.0/3.0)
    assert policy.get_average_value(0) == 0.0 # No incoming edges to 0

# Author: Ondrej Lukas, ondrej.lukas@aic.fel.cvut.cz
import numpy as np
import ruptures as rpt
from typing import Iterable
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import DBSCAN
from collections import defaultdict

def trajectory_str_representation(trajectory:Iterable)->str:
    if not trajectory:
        return ""
    parts = [f"[{trajectory[0].state}]"]
    for t in trajectory:
        parts.append(f"-{t.action}->[{t.next_state}]")
    return "".join(parts)

def compute_lambda_returns(trajectory:Iterable, gamma=0.99, lam=0.95):
    """
    Compute eligibility traces for a trajectory.
    Args:
        trajectory (Iterable): A trajectory, which is a list of transitions.
        gamma (float): Discount factor.
        lam (float): Lambda for eligibility traces.
    Returns:
        np.ndarray: λreturns for the trajectory (np.array).
    """
    T = len(trajectory)
    λ_ret = np.zeros(T)
    λ_ret[-1] = trajectory[-1].reward
    for t in reversed(range(T - 1)):
        λ_ret[t] = trajectory[t].reward + gamma * lam * λ_ret[t + 1]
    return λ_ret

def compute_eligibility_traces_sa(trajectory: Iterable, gamma=0.99, lam=0.95):
    e = defaultdict(float)
    traces = []

    for transition in trajectory:
        s, a = transition.state, transition.action

        # Decay all traces
        for key in e:
            e[key] *= gamma * lam

        # Bump trace for current state-action
        e[(s, a)] += 1.0

        # Store a snapshot
        traces.append(dict(e))  # shallow copy
    # Prune low values?
    return traces  # List of dicts: one per timestep

def compute_credit_per_step(trajectory:Iterable, gamma=0.99, lam=0.95):
    """
    Compute credit per step for a trajectory.
    Args:
        trajectory (Iterable): A trajectory, which is a list of transitions.
        gamma (float): Discount factor.
        lam (float): Lambda for eligibility traces.
    Returns:
        np.ndarray: Credit per step for the trajectory (np.array).
    """
    value = compute_lambda_returns(trajectory, gamma, lam)
    traces = compute_eligibility_traces_sa(trajectory, gamma, lam)
    credit = defaultdict(float)                   # final attribution per (s,a)
    for t, e_t in enumerate(traces):              # iterate over timesteps
        v = value[t]                              # scalar to propagate at step t
        for (s,a), elig in e_t.items():           # sparse over only active pairs
            credit[(s,a)] += elig * v 

    

def get_trajectory_action_surprises(trajectory:list, graph, epsilon=1e-8):
    """
    Computes the action surprise of a trajectory.
    """
    action_surprises = []
    for transition in trajectory:
        action_surprise = graph.compute_action_surprise(transition, epsilon)
        action_surprises.append(action_surprise)
    return action_surprises

def get_segment_features(seg_start:int, seg_end:int ,surprises:np.ndarray,rewards:np.ndarray, elegibility_traces:np.ndarray, trajectory:Iterable):
    """
    Computes the features for a segment.
    """
    features = {}
    features["et"] = np.mean(elegibility_traces[seg_start:seg_end])
    features["et_std"] = np.std(elegibility_traces[seg_start:seg_end])
    features["surprise"] = np.mean(surprises[seg_start:seg_end])
    #features["surprise_std"] = np.std(surprises[seg_start:seg_end])
    features["reward"] = np.mean(rewards[seg_start:seg_end])
    features["reward_std"] = np.std(rewards[seg_start:seg_end])
    features["coverage"] = (seg_end - seg_start)/len(trajectory)
    features["pos_start"] = seg_start/len(trajectory)
    features["pos_end"] = seg_end/len(trajectory)
    return features

def get_cluster_features(segments:Iterable):
    features = np.zeros([len(segments), len(segments[0][1].values())]) 
    for i,s in enumerate(segments):
        for j,x in enumerate(s[1].values()):
            features[i][j] = x
    ret = {}
    for i, k in enumerate(segments[0][1].keys()):
        ret[k] = {
            "mean":np.mean(features[:, i]),
            "std":np.std(features[:, i]),
            # "min":np.min(features[:, i]),
            # "max":np.max(features[:, i]),
            # "median":np.median(features[:, i]),
        }
    return ret
 
def cluster_segments(segments:Iterable):
    features = [list(s[1].values()) for s in segments]
    clustering = DBSCAN(eps=5, min_samples=2).fit(features)
    clusters = {}
    for cluster_id in np.unique(clustering.labels_).tolist():
        if cluster_id == -1:
            continue
        clusters[cluster_id] = []
        for i, s in enumerate(segments):
            if clustering.labels_[i] == cluster_id:
               clusters[cluster_id].append(s)
        print(f"Cluster {cluster_id}: {len(clusters[cluster_id])} segments, {len(set([x[0] for x in clusters[cluster_id]]))}({len(set([x[1].values() for x in clusters[cluster_id]]))}) unique segments")
        for s in set([x[0] for x in clusters[cluster_id]]):
            print(trajectory_str_representation(s))
        print("Features:")
        features = get_cluster_features(clusters[cluster_id])
        for k, v in features.items():
            print(f"{k}: {v}")
        print("#########################")
    print("-" * 50)
    return clusters

def get_motifs(trajectories, graph, epsilon=1e-12, penalty=2):
    """
    Computes the motifs in the transition graph.
    """
    motifs = {}
    motfif_candidates = []
    for t in trajectories:
        rewards = np.array([step.reward for step in t])
        lambda_ret = compute_lambda_returns(t)
        surprises = np.array(get_trajectory_action_surprises(t, graph, epsilon))
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(np.stack([lambda_ret, surprises, rewards], axis=1))
        algo = rpt.KernelCPD(kernel="rbf", min_size=1).fit(features_scaled)
        bkpts = algo.predict(pen=penalty)
        segments = [(0, bkpts[0])] + [(bkpts[i], bkpts[i+1]) for i in range(len(bkpts)-1)]
        for start, end in segments:
            m = tuple(t[start:end])
            motfif_candidates.append((m, get_segment_features(start, end, surprises, rewards, lambda_ret, t)))
            if m not in motifs:
                motifs[m] = 0
            motifs[m] += 1
        # add segment occurence to the motif features
        final_candidates = []
        for m_candidate, features in motfif_candidates:
            features["occurences"] = motifs[m]
            final_candidates.append((m_candidate, features))
    clusters = cluster_segments(final_candidates)
    return motifs, clusters
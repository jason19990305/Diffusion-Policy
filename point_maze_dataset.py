import torch
import numpy as np
import minari
from torch.utils.data import Dataset
import os

from utils.normalization import NumpyNormalizer

class PointMazeDataset(Dataset):
    def __init__(self, dataset_id="D4RL/pointmaze/large-v2", pred_horizon=16, obs_horizon=2):
        print(f"Loading dataset {dataset_id}...")
        self.dataset = minari.load_dataset(dataset_id, download=True)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        
        self.states = []
        self.actions = []
        
        # PointMaze Large-v3/v2 observations are dicts: 
        # 'observation': [pos_x, pos_y, vel_x, vel_y]
        # 'desired_goal': [goal_x, goal_y]
        
        for episode in self.dataset.iterate_episodes():
            # Combine 'observation' and 'desired_goal' into a single state vector
            # obs shape in dataset: (T, 4), goal shape: (T, 2) or single (2,)
            # Minari PointMaze large-v2 observation is (4,) and desired_goal is (2,)
            obs = episode.observations['observation'] # (T+1, 4)
            goal = episode.observations['desired_goal'] # (T+1, 2)
            acts = episode.actions # (T, 2)
            
            # Match lengths: actions are T, observations are T+1
            # We use state at t to predict action at t
            combined_state = np.concatenate([obs[:-1], goal[:-1]], axis=-1) # (T, 6)
            
            self.states.append(combined_state)
            self.actions.append(acts)
            
        # Flatten all episodes into one big array for normalization
        all_states = np.concatenate(self.states, axis=0)
        all_actions = np.concatenate(self.actions, axis=0)
        
        self.state_normalizer = NumpyNormalizer(all_states)
        self.action_normalizer = NumpyNormalizer(all_actions)
        
        # Normalize data
        self.normalized_states = [self.state_normalizer.normalize(s) for s in self.states]
        self.normalized_actions = [self.action_normalizer.normalize(a) for a in self.actions]
        
        # Indices for sequence sampling
        self.indices = []
        for i, traj in enumerate(self.normalized_states):
            # We need enough steps for obs and prediction
            # idx is the *current* time step t
            for t in range(len(traj)):
                self.indices.append((i, t))
                
        self.state_dim = 6 # [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y]
        self.action_dim = 2 # [force_x, force_y]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, t = self.indices[idx]
        
        states = self.normalized_states[traj_idx]
        actions = self.normalized_actions[traj_idx]
        traj_len = len(states)
        
        # 0. Perfect Alignment (Aligned Chunking)
        # Make observation and action sequences start at the same time step
        t_start = t - self.obs_horizon + 1
        
        # 1. Observation sequence: use np.clip for index clamping
        obs_indices = np.clip(np.arange(t_start, t_start + self.obs_horizon), 0, traj_len - 1)
        obs_seq = states[obs_indices] # shape: (obs_horizon, state_dim)
        
        # 2. Action sequence: natively aligned with obs_start
        action_indices = np.clip(np.arange(t_start, t_start + self.pred_horizon), 0, traj_len - 1)
        action_seq = actions[action_indices] # shape: (pred_horizon, action_dim)
            
        return {
            "obs": torch.from_numpy(obs_seq).float(),
            "action": torch.from_numpy(action_seq).float()
        }

if __name__ == "__main__":
    # Test dataset
    try:
        dataset = PointMazeDataset(pred_horizon=16, obs_horizon=2)
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        sample = dataset[0]
        print(f"Observation shape: {sample['obs'].shape}") # Expected: [2, 6]
        print(f"Action shape: {sample['action'].shape}")   # Expected: [16, 2]
    except Exception as e:
        print(f"Error: {e}")

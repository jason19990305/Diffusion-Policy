import torch
import numpy as np
from torch.utils.data import Dataset


from trajectory_plot import generate_trajectory_forward, generate_trajectory_reverse

# ---------------------------------
# Normalization Class
# ---------------------------------
class Normalization:
    # Min-Max Normalization
    def __init__(self, data: np.ndarray):
        self.min = np.min(data, axis=0) # [dim]
        self.max = np.max(data, axis=0) # [dim]
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-5

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min

# ---------------------------------
# Trajectory Dataset Class
# ---------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, pred_horizon, obs_horizon=8):
        
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        
        # 1. define dimension of action and state
        self.action_dim = 2
        self.state_dim = 2 # Here, state is just the (x, y) position
        
        # 2. generate the two trajectories
        self.traj_fwd_np = generate_trajectory_forward(num_steps=100)  # shape: (100, 2)
        self.traj_rev_np = generate_trajectory_reverse(num_steps=100)  # shape: (100, 2)
        
        # Initialize normalizer on the combined dataset to ensure scale is equal
        combined_np = np.concatenate([self.traj_fwd_np, self.traj_rev_np], axis=0)
        self.normalizer = Normalization(combined_np)
        
        # apply normalization
        self.traj_fwd_np = self.normalizer.normalize(self.traj_fwd_np)
        self.traj_rev_np = self.normalizer.normalize(self.traj_rev_np)
        
        # convert to torch tensors
        self.traj_fwd = torch.from_numpy(self.traj_fwd_np).float() 
        self.traj_rev = torch.from_numpy(self.traj_rev_np).float() 
        
        self.step_num = 100 # Steps per trajectory
        
        # 3. Calculate dataset length (100 steps * 2 modalities)
        self.length = self.step_num * 2
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # is forward trajectory or reverse trajectory
        is_forward = idx < self.step_num
        t = idx % self.step_num
        
        # select forward or reverse trajectory
        trajectory = self.traj_fwd if is_forward else self.traj_rev
        
        # 1. Observation sequence: length 'obs_horizon' ending at 't'
        obs_indices = torch.arange(t - self.obs_horizon + 1, t + 1)
        # Pad start of the episode by repeating the first frame
        obs_indices = torch.clamp(obs_indices, min=0, max=self.step_num - 1)
        obs_seq = trajectory[obs_indices]  # shape: (obs_horizon, state_dim)
        
        # 2. Action sequence: length 'pred_horizon' starting from 't + 1' 
        action_indices = torch.arange(t + 1, t + 1 + self.pred_horizon)
        # Pad end of the episode by repeating the last frame
        action_indices = torch.clamp(action_indices, min=0, max=self.step_num - 1)
        action_seq = trajectory[action_indices]  # shape: (pred_horizon, action_dim)
        
        return {
            "obs": obs_seq,
            "action": action_seq
        }

# --- Quick Test ---
if __name__ == "__main__":
    dataset = TrajectoryDataset(pred_horizon=16, obs_horizon=8)
    
    # Test idx = 0 (Start of episode, testing padding)
    sample_start = dataset[0]
    print("--- Testing idx = 0 ---")
    print("Obs shape:", sample_start["obs"].shape)       # Expected:[8, 2]
    print("First Obs:\n", sample_start["obs"])           # Should see repeated first states
    
    # Test idx = 95 (End of episode, testing action padding)
    sample_end = dataset[95]
    print("\n--- Testing idx = 95 ---")
    print("Action shape:", sample_end["action"].shape)   # Expected:[16, 2]
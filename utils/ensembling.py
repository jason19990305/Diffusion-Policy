import numpy as np
import torch

class NumpyTemporalEnsembler:
    """
    Temporal Ensembling for action smoothing across overlapping prediction horizons.
    Implemented in NumPy.
    """
    def __init__(self, pred_horizon: int, action_dim: int):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_sum = np.zeros((pred_horizon, action_dim))
        self.action_count = np.zeros((pred_horizon, 1))

    def update(self, predicted_action_seq: np.ndarray):
        """Add a new predicted sequence to the ensemble buffer."""
        length = min(len(predicted_action_seq), self.pred_horizon)
        self.action_sum[:length] += predicted_action_seq[:length]
        self.action_count[:length] += 1

    def get_and_shift_actions(self, n_actions: int) -> np.ndarray:
        """Compute average actions and shift the buffer window forward."""
        # Prevent division by zero
        counts = np.clip(self.action_count[:n_actions], a_min=1, a_max=None)
        avg_actions = self.action_sum[:n_actions] / counts
        
        # Update and shift the buffer
        new_sum = np.zeros_like(self.action_sum)
        new_count = np.zeros_like(self.action_count)
        
        if self.pred_horizon > n_actions:
            new_sum[:-n_actions] = self.action_sum[n_actions:]
            new_count[:-n_actions] = self.action_count[n_actions:]
            
        self.action_sum = new_sum
        self.action_count = new_count
        
        return avg_actions

class TensorTemporalEnsembler:
    """
    Temporal Ensembling for action smoothing across overlapping prediction horizons.
    Implemented in PyTorch.
    """
    def __init__(self, pred_horizon: int, action_dim: int):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_sum = torch.zeros((pred_horizon, action_dim))
        self.action_count = torch.zeros((pred_horizon, 1))

    def update(self, predicted_action_seq: torch.Tensor):
        """Add a new predicted sequence to the ensemble buffer."""
        length = min(len(predicted_action_seq), self.pred_horizon)
        self.action_sum[:length] += predicted_action_seq[:length]
        self.action_count[:length] += 1

    def get_and_shift_actions(self, n_actions: int) -> torch.Tensor:
        """Compute average actions and shift the buffer window forward."""
        counts = torch.clamp(self.action_count[:n_actions], min=1)
        avg_actions = self.action_sum[:n_actions] / counts
        
        new_sum = torch.zeros_like(self.action_sum)
        new_count = torch.zeros_like(self.action_count)
        
        if self.pred_horizon > n_actions:
            new_sum[:-n_actions] = self.action_sum[n_actions:]
            new_count[:-n_actions] = self.action_count[n_actions:]
            
        self.action_sum = new_sum
        self.action_count = new_count
        return avg_actions

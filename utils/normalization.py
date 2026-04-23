import numpy as np
import torch

class NumpyNormalizer:
    """
    Standard Min-Max Normalizer implemented in NumPy.
    """
    def __init__(self, data: np.ndarray):
        # data: [N, dim]
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-5

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return (x_norm + 1.0) / 2.0 * self.range + self.min

class TensorNormalizer:
    """
    Standard Min-Max Normalizer implemented strictly in PyTorch.
    This avoids expensive Numpy <-> Tensor conversions during __getitem__.
    """
    def __init__(self, min_val, max_val):
        self.min = torch.tensor(min_val, dtype=torch.float32)
        self.max = torch.tensor(max_val, dtype=torch.float32)
        self.range = torch.clamp(self.max - self.min, min=1e-5)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return (x_norm + 1.0) / 2.0 * self.range + self.min

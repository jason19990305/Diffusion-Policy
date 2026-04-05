import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import warnings

# Suppress torchvision warnings regarding video decoding
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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

class AlohaDataset(Dataset):
    """
    Optimized ALOHA Dataset (Aligned Action Chunking) for Diffusion Policy.
    """
    DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

    def __init__(
        self,
        pred_horizon: int = 16,
        obs_horizon: int = 4,
        image_size: int = 128,
        cache_dir: str = "cache",
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.image_size = image_size

        self.lerobot_dataset = LeRobotDataset(self.DATASET_ID, video_backend="pyav")
        hf_data = self.lerobot_dataset.hf_dataset
        
        self.cam_key = next(k for k in self.lerobot_dataset.features if k.startswith("observation.images."))
        
        print("[AlohaDataset] Analyzing episode boundaries...")
        ep_ids = torch.as_tensor(hf_data["episode_index"])
        diff = torch.diff(ep_ids)
        change_indices = torch.where(diff != 0)[0] + 1
        
        self.ep_boundaries = torch.cat([torch.tensor([0]), change_indices, torch.tensor([len(ep_ids)])])
        self.frame_to_ep_id = ep_ids
        print(f"[AlohaDataset] Detected {len(self.ep_boundaries)-1} episodes.")

        print("[AlohaDataset] Caching States & Actions to RAM...")
        self.cached_states = torch.stack([torch.as_tensor(s) for s in hf_data["observation.state"]]).to(torch.float32)
        self.cached_actions = torch.stack([torch.as_tensor(a) for a in hf_data["action"]]).to(torch.float32)
        
        stats = self.lerobot_dataset.meta.stats
        self.state_normalizer = TensorNormalizer(stats["observation.state"]["min"], stats["observation.state"]["max"])
        self.action_normalizer = TensorNormalizer(stats["action"]["min"], stats["action"]["max"])

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"aloha_single_img_obs{obs_horizon}_s{image_size}.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading single-cam cache via mmap: {self.cache_path}")
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()
            
        self.state_dim = self.cached_states.shape[-1]
        self.action_dim = self.cached_actions.shape[-1]
        print(f"[AlohaDataset] Ready! state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _create_image_cache(self):
        print("[AlohaDataset] Cache not found. Creating FP16 image cache...")
        n = len(self.lerobot_dataset)
        resize = T.Resize((self.image_size, self.image_size), antialias=True)
        
        self.cached_images = torch.zeros((n, 3, self.image_size, self.image_size), dtype=torch.float16)

        for i in tqdm(range(n), desc="Processing Frames"):
            img = self.lerobot_dataset[i][self.cam_key]
            if img.ndim == 4:
                img = img[-1]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype != torch.float32:
                img = img.float()
            self.cached_images[i] = resize(img).half()

        print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
        torch.save(self.cached_images, self.cache_path)

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves synchronized observation/action windows."""
        ep_id = self.frame_to_ep_id[idx].item()
        ep_start = self.ep_boundaries[ep_id].item()
        ep_end = self.ep_boundaries[ep_id + 1].item()

        # ----------------------------------------------------
        # 1. Perfect Alignment (Aligned Chunking)
        # ----------------------------------------------------
        # Make observation and action sequences start at the same time step (t - obs_horizon + 1)
        t_start = idx - self.obs_horizon + 1
        
        obs_steps = torch.arange(t_start, t_start + self.obs_horizon)
        act_steps = torch.arange(t_start, t_start + self.pred_horizon)

        # ----------------------------------------------------
        # 2. Automatic Padding for Boundary Cases (Clamp)
        # ----------------------------------------------------
        # Clamp limits the indices to not fall outside [ep_start, ep_end - 1]
        # This automatically pads out-of-bounds indices by replicating the first/last frame
        obs_steps_clamped = torch.clamp(obs_steps, min=ep_start, max=ep_end - 1)
        act_steps_clamped = torch.clamp(act_steps, min=ep_start, max=ep_end - 1)

        # ----------------------------------------------------
        # 3. Retrieve Data and Apply Augmentations (All Tensors)
        # ----------------------------------------------------
        raw_state = self.cached_states[obs_steps_clamped]
        img_seq = self.cached_images[obs_steps_clamped].float()
        raw_action = self.cached_actions[act_steps_clamped]

        # Apply Consistent Data Augmentation across the entire observation window
        import torch.nn.functional as F
        
        # --- 3.1 Consistent Brightness and Contrast ---
        brightness = 1.0 + float(torch.empty(1).uniform_(-0.2, 0.2))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.2, 0.2))
        
        img_seq = img_seq * brightness
        img_seq = torch.clamp((img_seq - 0.5) * contrast + 0.5, 0.0, 1.0)
        
        # --- 3.2 Consistent Random Translation (Translation via Padding and Cropping) ---
        pad = 4
        img_seq_padded = F.pad(img_seq, (pad, pad, pad, pad), mode='replicate')
        
        # Same random crop coordinates for all images in this sequence
        dx = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        dy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        
        img_seq = img_seq_padded[..., dy:dy + self.image_size, dx:dx + self.image_size]

        return {
            "obs":    self.state_normalizer.normalize(raw_state),
            "image":  img_seq, 
            "action": self.action_normalizer.normalize(raw_action),
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = AlohaDataset(pred_horizon=16, obs_horizon=4, image_size=128)
    sample = dataset[len(dataset) // 2]
    print(f"obs shape:    {list(sample['obs'].shape)}")    
    print(f"action shape: {list(sample['action'].shape)}")
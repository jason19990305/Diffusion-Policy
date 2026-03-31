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

class Normalization:
    """Standard Min-Max Normalization helper."""
    def __init__(self, min_val, max_val):
        self.min = min_val.copy().astype(np.float32)
        self.max = max_val.copy().astype(np.float32)
        self.range = np.where((self.max - self.min) == 0, 1e-5, self.max - self.min)

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min

class AlohaDataset(Dataset):
    """
    Optimized ALOHA Dataset for LeRobot v0.5.x.
    Fixed image dimension handling and manual episode boundary detection.
    """
    DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

    def __init__(
        self,
        pred_horizon:    int  = 16,
        obs_horizon:     int  = 4,
        image_size:      int  = 128,
        cache_dir:       str  = "cache",
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon  = obs_horizon
        self.image_size   = image_size

        # 1. Load basic dataset structure
        self.lerobot_dataset = LeRobotDataset(self.DATASET_ID, video_backend="pyav")
        self.camera_keys = [k for k in self.lerobot_dataset.features if k.startswith("observation.images.")]
        hf_data = self.lerobot_dataset.hf_dataset

        # 2. Manual Episode Boundary Detection (Bypassing unstable API attributes)
        # We calculate boundaries from the 'episode_index' column directly.
        print("[AlohaDataset] Analyzing episode boundaries...")
        ep_ids = np.array(hf_data["episode_index"])
        diff = np.diff(ep_ids)
        change_indices = np.where(diff != 0)[0] + 1
        self.episode_data_index = np.concatenate([[0], change_indices, [len(ep_ids)]])
        self.frame_to_episode_id = torch.from_numpy(ep_ids)
        print(f"[AlohaDataset] Detected {len(self.episode_data_index)-1} episodes.")

        # 3. Fast Cache States & Actions to RAM
        print("[AlohaDataset] Fast Caching States & Actions...")
        self.cached_states = torch.from_numpy(np.array(hf_data["observation.state"])).float()
        self.cached_actions = torch.from_numpy(np.array(hf_data["action"])).float()
        
        # 4. Normalization Statistics
        stats = self.lerobot_dataset.meta.stats
        self.state_normalizer  = Normalization(np.array(stats["observation.state"]["min"]), np.array(stats["observation.state"]["max"]))
        self.action_normalizer = Normalization(np.array(stats["action"]["min"]), np.array(stats["action"]["max"]))

        # 5. Image Cache (Memory-Mapped for H100 efficiency)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"aloha_img_obs{obs_horizon}_s{image_size}.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading image cache via mmap: {self.cache_path}")
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()
            
        # --- Add these two lines at the end of __init__ ---
        self.state_dim = self.cached_states.shape[-1]   # Should be 14 for ALOHA
        self.action_dim = self.cached_actions.shape[-1] # Should be 14 for ALOHA
        print(f"[AlohaDataset] Initialized with state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _create_image_cache(self):
        """Decoding and resizing frames. Stored as FP16 to minimize disk/RAM usage."""
        print("[AlohaDataset] No cache found. Creating image cache...")
        n = len(self.lerobot_dataset)
        num_cams = len(self.camera_keys)
        resize = T.Resize((self.image_size, self.image_size), antialias=True)
        
        # Pre-allocate (N, Cams, C, H, W)
        self.cached_images = torch.zeros((n, num_cams, 3, self.image_size, self.image_size), dtype=torch.float16)

        for i in tqdm(range(n), desc="Decoding Video Frames"):
            item = self.lerobot_dataset[i]
            for c_idx, k in enumerate(self.camera_keys):
                # FIXED: Correctly handle image dimensions
                img = item[k] # Shape should be (C, H, W) or (T, C, H, W)
                
                # If the dataset returns a sequence (T, C, H, W), take the last frame
                if img.ndim == 4:
                    img = img[-1]
                
                # Ensure img is (C, H, W) before resizing
                if img.ndim == 2: # (H, W) -> (1, H, W)
                    img = img.unsqueeze(0)
                
                if img.dtype != torch.float32: img = img.float()
                if img.max() > 1.5: img /= 255.0 # Normalizing to [0, 1]
                
                # Resizing and converting to FP16
                self.cached_images[i, c_idx] = resize(img).half()

        print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
        torch.save(self.cached_images, self.cache_path)

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves synchronized observation/action windows."""
        ep_id = self.frame_to_episode_id[idx]
        start_idx = self.episode_data_index[ep_id]
        end_idx = self.episode_data_index[ep_id + 1]

        # 1. Observation Slicing (with Episode Start Padding)
        s_lookback = max(start_idx, idx - self.obs_horizon + 1)
        img_seq = self.cached_images[s_lookback : idx + 1]
        raw_state = self.cached_states[s_lookback : idx + 1]
        
        if img_seq.shape[0] < self.obs_horizon:
            pad_len = self.obs_horizon - img_seq.shape[0]
            img_seq = torch.cat([img_seq[0:1].expand(pad_len, -1, -1, -1, -1), img_seq], dim=0)
            raw_state = torch.cat([raw_state[0:1].expand(pad_len, -1), raw_state], dim=0)

        # 2. Action Slicing (with Episode End Padding)
        e_lookforward = min(end_idx, idx + self.pred_horizon)
        raw_action = self.cached_actions[idx : e_lookforward]
        
        if raw_action.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - raw_action.shape[0]
            raw_action = torch.cat([raw_action, raw_action[-1:].expand(pad_len, -1)], dim=0)

        # 3. Apply Normalization
        state_norm  = self.state_normalizer.normalize(raw_state.numpy())
        action_norm = self.action_normalizer.normalize(raw_action.numpy())

        return {
            "obs":    torch.from_numpy(state_norm).float(),
            "image":  img_seq.float(), 
            "action": torch.from_numpy(action_norm).float(),
        }
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class Normalization:
    """Standard Min-Max Normalization to [-1, 1]"""
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
    Robust ALOHA Dataset for LeRobot v0.5.0+.
    Uses manual boundary detection to avoid unstable API attributes.
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

        # 1. Load Dataset Structure
        self.lerobot_dataset = LeRobotDataset(self.DATASET_ID, video_backend="pyav")
        self.camera_keys = [k for k in self.lerobot_dataset.features if k.startswith("observation.images.")]
        
        # Access the underlying HuggingFace dataset
        hf_data = self.lerobot_dataset.hf_dataset

        # ------------------------------------------------------------------
        # FIX: Manual Episode Boundary Detection (Bulletproof Method)
        # Instead of relying on self.lerobot_dataset.episode_data_index, 
        # we calculate it directly from the 'episode_index' column.
        # ------------------------------------------------------------------
        print("[AlohaDataset] Analyzing episode boundaries...")
        ep_ids = np.array(hf_data["episode_index"]) # e.g., [0,0,0,1,1,2,2,2...]
        
        # Find where the episode ID changes
        diff = np.diff(ep_ids)
        change_indices = np.where(diff != 0)[0] + 1
        
        # Construct boundary array: [start_of_ep0, start_of_ep1, ..., total_length]
        self.episode_data_index = np.concatenate([[0], change_indices, [len(ep_ids)]])
        self.frame_to_episode_id = torch.from_numpy(ep_ids)
        
        print(f"[AlohaDataset] Detected {len(self.episode_data_index)-1} episodes.")
        # ------------------------------------------------------------------

        # 2. Fast Cache States & Actions to RAM
        print("[AlohaDataset] Fast Caching States & Actions...")
        self.cached_states = torch.from_numpy(np.array(hf_data["observation.state"])).float()
        self.cached_actions = torch.from_numpy(np.array(hf_data["action"])).float()
        
        # 3. Normalization Stats
        stats = self.lerobot_dataset.meta.stats
        self.state_normalizer  = Normalization(np.array(stats["observation.state"]["min"]), np.array(stats["observation.state"]["max"]))
        self.action_normalizer = Normalization(np.array(stats["action"]["min"]), np.array(stats["action"]["max"]))

        # 4. Image Cache (Memory-Mapped)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"aloha_img_obs{obs_horizon}_s{image_size}.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading image cache via mmap: {self.cache_path}")
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()

    def _create_image_cache(self):
        """Decoding and resizing frames to float16 cache file."""
        print("[AlohaDataset] No cache found. Creating image cache...")
        n = len(self.lerobot_dataset)
        num_cams = len(self.camera_keys)
        resize = T.Resize((self.image_size, self.image_size), antialias=True)
        
        self.cached_images = torch.zeros((n, num_cams, 3, self.image_size, self.image_size), dtype=torch.float16)

        for i in tqdm(range(n), desc="Decoding Video Frames"):
            item = self.lerobot_dataset[i]
            for c_idx, k in enumerate(self.camera_keys):
                img = item[k][-1] # Take the last frame of the window
                if img.dtype != torch.float32: img = img.float()
                if img.max() > 1.5: img /= 255.0
                self.cached_images[i, c_idx] = resize(img).half()

        print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
        torch.save(self.cached_images, self.cache_path)

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Fetch synchronized data windows within episode boundaries."""
        ep_id = self.frame_to_episode_id[idx]
        start_idx = self.episode_data_index[ep_id]
        end_idx = self.episode_data_index[ep_id + 1]

        # --- 1. Observation Window ---
        s_lookback = max(start_idx, idx - self.obs_horizon + 1)
        img_seq = self.cached_images[s_lookback : idx + 1]
        raw_state = self.cached_states[s_lookback : idx + 1]
        
        # Padding for Episode Start
        if img_seq.shape[0] < self.obs_horizon:
            pad_len = self.obs_horizon - img_seq.shape[0]
            img_seq = torch.cat([img_seq[0:1].expand(pad_len, -1, -1, -1, -1), img_seq], dim=0)
            raw_state = torch.cat([raw_state[0:1].expand(pad_len, -1), raw_state], dim=0)

        # --- 2. Action Window ---
        e_lookforward = min(end_idx, idx + self.pred_horizon)
        raw_action = self.cached_actions[idx : e_lookforward]
        
        # Padding for Episode End
        if raw_action.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - raw_action.shape[0]
            raw_action = torch.cat([raw_action, raw_action[-1:].expand(pad_len, -1)], dim=0)

        # --- 3. Finalize ---
        state_norm  = self.state_normalizer.normalize(raw_state.numpy())
        action_norm = self.action_normalizer.normalize(raw_action.numpy())

        return {
            "obs":    torch.from_numpy(state_norm).float(),
            "image":  img_seq.float(), 
            "action": torch.from_numpy(action_norm).float(),
        }
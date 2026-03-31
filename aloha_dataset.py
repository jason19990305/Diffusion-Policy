import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import warnings

# Suppress torchvision video warnings (LeRobot handles video decoding via PyAV)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class Normalization:
    """Helper class for Min-Max normalization to [-1, 1] range."""
    def __init__(self, min_val, max_val):
        self.min = min_val.copy().astype(np.float32)
        self.max = max_val.copy().astype(np.float32)
        # Prevent division by zero for constant dimensions
        self.range = np.where((self.max - self.min) == 0, 1e-5, self.max - self.min)

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min

class AlohaDataset(Dataset):
    """
    Optimized Dataset class for ALOHA Diffusion Policy (LeRobot v0.5.0 compatible).
    Features RAM caching for states/actions and Memory-Mapped (mmap) image caching.
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

        # 1. Initialize LeRobot dataset structure
        self.lerobot_dataset = LeRobotDataset(self.DATASET_ID, video_backend="pyav")
        self.camera_keys = [k for k in self.lerobot_dataset.features if k.startswith("observation.images.")]
        
        # Access underlying HuggingFace dataset for high-speed batch access
        hf_data = self.lerobot_dataset.hf_dataset

        # 2. Handle Episode Indices (Compatibility fix for LeRobot v0.5.0)
        # episode_data_index tracks the start/end frames for each episode
        if hasattr(self.lerobot_dataset, "episode_data_index"):
            self.episode_data_index = self.lerobot_dataset.episode_data_index
        else:
            self.episode_data_index = self.lerobot_dataset.meta.episode_data_index

        # Maps each frame index to its corresponding episode ID
        self.frame_to_episode_id = torch.from_numpy(np.array(hf_data["episode_index"]))

        # 3. Fast RAM Caching: States and Actions
        # Fetching entire columns at once is ~100x faster than indexed looping
        print("[AlohaDataset] Fast Caching States & Actions to RAM...")
        self.cached_states = torch.from_numpy(np.array(hf_data["observation.state"])).float()
        self.cached_actions = torch.from_numpy(np.array(hf_data["action"])).float()
        
        # 4. Normalization Statistics
        stats = self.lerobot_dataset.meta.stats
        self.state_normalizer  = Normalization(np.array(stats["observation.state"]["min"]), np.array(stats["observation.state"]["max"]))
        self.action_normalizer = Normalization(np.array(stats["action"]["min"]), np.array(stats["action"]["max"]))

        # 5. Optimized Image Cache (Memory Mapping)
        # Using mmap=True allows near-instant loading and efficient RAM usage on H100
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"aloha_img_obs{obs_horizon}_s{image_size}.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading image cache via mmap: {self.cache_path}")
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()

    def _create_image_cache(self):
        """One-time video decoding and resizing. Saves as float16 to optimize storage."""
        print("[AlohaDataset] No cache found. Decoding videos (this may take a few minutes)...")
        n = len(self.lerobot_dataset)
        num_cams = len(self.camera_keys)
        
        resize = T.Resize((self.image_size, self.image_size), antialias=True)
        
        # Pre-allocate tensor (N, Num_Cams, C, H, W) in FP16 to halve disk/RAM footprint
        self.cached_images = torch.zeros((n, num_cams, 3, self.image_size, self.image_size), dtype=torch.float16)

        for i in tqdm(range(n), desc="Decoding Frames"):
            item = self.lerobot_dataset[i]
            for c_idx, k in enumerate(self.camera_keys):
                # Retrieve last frame (T=1) from the returned sequence
                img = item[k][-1] 
                if img.dtype != torch.float32: img = img.float()
                if img.max() > 1.5: img /= 255.0 # Scale to [0, 1]
                self.cached_images[i, c_idx] = resize(img).half()

        print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
        torch.save(self.cached_images, self.cache_path)

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves synchronized sequence of images, states, and actions."""
        # Get episode boundaries to prevent cross-episode data leakage
        ep_id = self.frame_to_episode_id[idx]
        start_idx = self.episode_data_index[ep_id]
        end_idx = self.episode_data_index[ep_id + 1]

        # 1. Observation Window (Lookback)
        # If the window goes before episode start, we clip and then pad later
        s_lookback = max(start_idx, idx - self.obs_horizon + 1)
        
        # Slicing tensors is extremely efficient
        img_seq = self.cached_images[s_lookback : idx + 1]
        raw_state = self.cached_states[s_lookback : idx + 1]
        
        # Handle Padding for Episode Start (Repeat first frame/state)
        if img_seq.shape[0] < self.obs_horizon:
            pad_len = self.obs_horizon - img_seq.shape[0]
            
            img_padding = img_seq[0:1].expand(pad_len, -1, -1, -1, -1)
            img_seq = torch.cat([img_padding, img_seq], dim=0)
            
            state_padding = raw_state[0:1].expand(pad_len, -1)
            raw_state = torch.cat([state_padding, raw_state], dim=0)

        # 2. Action Window (Lookforward)
        # Cannot predict beyond the current episode
        e_lookforward = min(end_idx, idx + self.pred_horizon)
        raw_action = self.cached_actions[idx : e_lookforward]
        
        # Handle Padding for Episode End (Repeat last action)
        if raw_action.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - raw_action.shape[0]
            action_padding = raw_action[-1:].expand(pad_len, -1)
            raw_action = torch.cat([raw_action, action_padding], dim=0)

        # 3. Final Normalization and Conversion
        state_norm  = self.state_normalizer.normalize(raw_state.numpy())
        action_norm = self.action_normalizer.normalize(raw_action.numpy())

        return {
            "obs":    torch.from_numpy(state_norm).float(),
            "image":  img_seq.float(), # Convert FP16 cache back to FP32 for model
            "action": torch.from_numpy(action_norm).float(),
        }
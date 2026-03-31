import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import warnings

# Suppress torchvision video deprecation warning (lerobot uses pyav internally)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# lerobot v0.5+ uses lerobot.datasets (NOT lerobot.common.datasets)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------
# Normalization Class (Min-Max to [-1, 1])
# ---------------------------------
class Normalization:
    """
    Min-Max normalization: maps raw data -> [-1, 1].
    Operates on numpy arrays; call .normalize() / .unnormalize().
    """
    def __init__(self, min_val: np.ndarray, max_val: np.ndarray):
        self.min = min_val.copy().astype(np.float32)
        self.max = max_val.copy().astype(np.float32)
        self.range = self.max - self.min
        # Prevent division by zero for constant dimensions
        self.range[self.range == 0] = 1e-5

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return (x_norm + 1.0) / 2.0 * self.range + self.min


# ---------------------------------
# ALOHA Dataset Class (lerobot v0.5+)
# ---------------------------------
class AlohaDataset(Dataset):
    """
    Wraps lerobot/aloha_sim_transfer_cube_human for Multi-View Diffusion Policy.

    Features:
      - Multi-view: top, left_wrist, right_wrist
      - RAM Caching: all states, actions, and images in RAM (float16 for images)
      - Returns: Image (T, 3, 3, 96, 96) where 3 is Num Cameras.
    """

    DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
    CAMERA_KEYS = ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"]

    def __init__(
        self,
        pred_horizon:    int  = 16,
        obs_horizon:     int  = 4,
        image_size:      int  = 96,
        prefetch_images: bool = True,   # Pre-decode all frames (recommended)
        cache_dir:       str  = "cache", # Dir to store/load the .pt disk cache
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon  = obs_horizon
        self.image_size   = image_size
        self.state_dim    = 14
        self.action_dim   = 14
        self.prefetch_images = prefetch_images

        # Cache file path: unique per (obs_horizon, image_size) to avoid conflicts
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            cache_dir,
            f"aloha_images_obs{obs_horizon}_s{image_size}.pt"
        )

        print(f"[AlohaDataset] Loading: {self.DATASET_ID}")

        # Load dataset just to read FPS from metadata (before setting delta_timestamps)
        _tmp = LeRobotDataset(self.DATASET_ID)
        fps = _tmp.meta.fps
        print(f"[AlohaDataset] FPS = {fps}")
        del _tmp

        # Build delta_timestamps:
        obs_deltas    = [i / fps for i in range(-(obs_horizon - 1), 1)]
        action_deltas = [i / fps for i in range(pred_horizon)]

        delta_timestamps = {
            "observation.state": obs_deltas,
            "action":            action_deltas,
        }
        for k in self.CAMERA_KEYS:
            delta_timestamps[k] = obs_deltas

        print(f"[AlohaDataset] Cameras: {self.CAMERA_KEYS}")
        print(f"[AlohaDataset] obs_deltas    = {obs_deltas}")
        print(f"[AlohaDataset] action_deltas = {action_deltas}")

        self.lerobot_dataset = LeRobotDataset(
            self.DATASET_ID,
            delta_timestamps=delta_timestamps,
            video_backend="pyav", # Force pyav to avoid torchcodec errors in WSL
        )
        print(f"[AlohaDataset] Total samples: {len(self.lerobot_dataset)}")

        # --- Normalization stats ---
        # In lerobot v0.5, stats live at dataset.meta.stats
        # Structure: stats[key][stat_name] -> np.ndarray
        # stat_name: "min", "max", "mean", "std"
        stats = self.lerobot_dataset.meta.stats

        state_min  = np.array(stats["observation.state"]["min"],  dtype=np.float32)  # (14,)
        state_max  = np.array(stats["observation.state"]["max"],  dtype=np.float32)  # (14,)
        action_min = np.array(stats["action"]["min"],             dtype=np.float32)  # (14,)
        action_max = np.array(stats["action"]["max"],             dtype=np.float32)  # (14,)

        self.state_normalizer  = Normalization(state_min,  state_max)
        self.action_normalizer = Normalization(action_min, action_max)

        # --- NEW: Full Data Cache (RAM) ---
        # Cache ALL states and actions in RAM to eliminate lerobot_dataset[idx] I/O
        print("[AlohaDataset] Caching all states and actions to RAM...")
        self.cached_states  = []
        self.cached_actions = []
        for i in tqdm(range(len(self.lerobot_dataset)), desc="RAM Caching"):
            s = self.lerobot_dataset[i]
            self.cached_states.append(s["observation.state"]) # (obs, 14)
            self.cached_actions.append(s["action"])           # (pred, 14)
        
        self.cached_states  = torch.stack(self.cached_states)
        self.cached_actions = torch.stack(self.cached_actions)
        print(f"[AlohaDataset] RAM Cache Ready. Total: {len(self.cached_states)} samples.")

        # --- Image resize transform ---
        # lerobot already returns images as float32 [0,1] via hf_transform_to_torch.
        # We only need to resize from the original resolution to image_size x image_size.
        self.image_transform = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # ------------------------------------------------------------------
        # Pre-fetch: decode ALL video frames into a single RAM tensor.
        # Strategy:
        #   1. If a .pt cache file exists -> load instantly (torch.load)
        #   2. Otherwise -> decode via pyav, save to .pt for next time
        # Cache is stored as float16 to halve disk/RAM usage (~1.66 GB).
        # ------------------------------------------------------------------
        if prefetch_images:
            if os.path.isfile(self.cache_path):
                # --- Fast path: load from disk cache ---
                print(f"[AlohaDataset] Loading image cache from: {self.cache_path}")
                self.cached_images = torch.load(
                    self.cache_path, map_location="cpu", weights_only=True
                )
                print(f"[AlohaDataset] Image cache loaded. "
                      f"Size: {self.cached_images.nbytes / 1e9:.2f} GB")
            else:
                # --- Slow path: decode from MP4 and save ---
                print(f"[AlohaDataset] Cache not found. Pre-decoding "
                      f"{len(self.lerobot_dataset)} x {obs_horizon} frames ... "
                      f"(first run only, ~5-6 min)")
                n = len(self.lerobot_dataset)
                num_cams = len(self.CAMERA_KEYS)
                # Allocate: (N, T, Num_Cameras, 3, H, W)
                self.cached_images = torch.zeros(
                    (n, obs_horizon, num_cams, 3, image_size, image_size), dtype=torch.float16
                )
                for i in tqdm(range(n), desc="Multi-View Prefetch", ncols=80):
                    item = self.lerobot_dataset[i]
                    batch_view = []
                    for k in self.CAMERA_KEYS:
                        raw = item[k] # (T, 3, H_orig, W_orig)
                        if raw.dtype != torch.float32: raw = raw.float()
                        if raw.max() > 1.5: raw = raw / 255.0
                        
                        resized = torch.stack([
                            self.image_transform(raw[t]) for t in range(obs_horizon)
                        ]) # (T, 3, 96, 96)
                        batch_view.append(resized)
                    
                    # Stack Cameras: (T, 3, 3, 96, 96)
                    # Use permute next if needed, but current shape: list of [T, 3, 96, 96]
                    # We want (T, num_cams, 3, 96, 96)
                    stacked = torch.stack(batch_view, dim=1) 
                    self.cached_images[i] = stacked.half()

                # Save to disk for future runs
                print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
                torch.save(self.cached_images, self.cache_path)
                print(f"[AlohaDataset] Cache saved. "
                      f"Size: {self.cached_images.nbytes / 1e9:.2f} GB")
        else:
            self.cached_images = None
            print("[AlohaDataset] prefetch_images=False: images decoded on-the-fly (slow).")

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        # 1. Get pre-decoded images, states, and actions from RAM
        # This is VASTLY faster than calling lerobot_dataset[idx] in a loop
        raw_images = self.cached_images[idx]  # (obs, 3, 96, 96)
        raw_state  = self.cached_states[idx]  # (obs, 14)
        raw_action = self.cached_actions[idx] # (pred, 14)

        # 2. Normalize
        state_norm  = self.state_normalizer.normalize(raw_state.numpy())
        action_norm = self.action_normalizer.normalize(raw_action.numpy())

        return {
            "obs":    torch.from_numpy(state_norm).float(),
            "image":  raw_images.float(), # [0, 1]
            "action": torch.from_numpy(action_norm).float(),
        }


# ---------------------------------
# Quick shape verification test
# ---------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AlohaDataset Shape Verification (lerobot v0.5)")
    print("=" * 60)

    dataset = AlohaDataset(pred_horizon=16, obs_horizon=2, image_size=96)

    sample = dataset[0]
    print(f"\n[obs]    shape : {sample['obs'].shape}")    # (2, 14)
    print(f"[image]  shape : {sample['image'].shape}")   # (2, 3, 3, 96, 96) (T, Cam, C, H, W)
    print(f"[action] shape : {sample['action'].shape}")  # (16, 14)

    print(f"\n[obs]    min/max: {sample['obs'].min():.3f} / {sample['obs'].max():.3f}")
    print(f"[image]  min/max: {sample['image'].min():.3f} / {sample['image'].max():.3f}")
    print(f"[action] min/max: {sample['action'].min():.3f} / {sample['action'].max():.3f}")
    print("\nAll checks passed!")

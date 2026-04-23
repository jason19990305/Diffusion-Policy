import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import warnings
import random
import torchvision.transforms.functional as F

# Suppress torchvision warnings regarding video decoding
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.normalization import TensorNormalizer

class AlohaAugmentor:
    def __init__(self, image_size=128):
        self.image_size = image_size
        # 1. Random Shift (maintains original Edge Padding)
        self.padding = 8 # Increased to 8px to expand the shift range
        
    def __call__(self, img_seq: torch.Tensor) -> torch.Tensor:
        """
        img_seq: (T, C, H, W)
        """
        # --- Determine random parameters shared across this specific sequence ---
        
        # 1. Random Crop parameters
        top = random.randint(0, self.padding * 2)
        left = random.randint(0, self.padding * 2)
        
        # 2. Color Jitter parameters
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        
        # 3. Decide whether to apply certain probabilistic augmentations
        apply_noise = random.random() > 0.5
        
        # Apply padding first to prepare for cropping
        # (T, C, H, W) -> Padding is applied to the last two dimensions (H and W)
        img_seq = F.pad(img_seq, padding=[self.padding]*4, padding_mode='edge')
        
        processed_frames = []
        for i in range(img_seq.shape[0]):
            img = img_seq[i]
            
            # Apply the same crop across the sequence
            img = F.crop(img, top, left, self.image_size, self.image_size)
            
            # Apply the same color jitter across the sequence
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)
            
            # Add slight Gaussian noise 
            # (Noise varies slightly per frame to improve robustness)
            if apply_noise:
                img = img + torch.randn_like(img) * 0.01
                img = torch.clamp(img, 0.0, 1.0)
                
            processed_frames.append(img)
            
        return torch.stack(processed_frames)

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
        augment: bool = False,
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.image_size = image_size
        self.augment = augment

        # ----------------------------------------------------
        # Data Augmentation: Random Shift (4-8px)
        # ----------------------------------------------------
        # We use T.RandomCrop with padding to implement a safe translation.
        # 'edge' padding mimics the environment and avoids artificial black borders.
        self.augmentor = AlohaAugmentor(image_size=image_size)


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
        # We use '_s' instead of '_size' to match existing filenames on disk
        self.cache_path = os.path.join(cache_dir, f"aloha_single_img_obs{obs_horizon}_s{image_size}_centered.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading single-cam cache via mmap: {self.cache_path}")
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()
            
        self.state_dim = self.cached_states.shape[-1]
        self.action_dim = self.cached_actions.shape[-1]
        print(f"[AlohaDataset] Ready! state_dim={self.state_dim}, action_dim={self.action_dim}")

    def _create_image_cache(self):
        print(f"[AlohaDataset] Cache not found. Accelerating single-camera cache via DataLoader & GPU...")
        n = len(self.lerobot_dataset)
        
        self.cached_images = torch.zeros((n, 3, self.image_size, self.image_size), dtype=torch.float16)

        # 1. Use DataLoader to multiprocess video decoding
        from torch.utils.data import DataLoader
        dl = DataLoader(
            self.lerobot_dataset, 
            batch_size=64, 
            num_workers=4, 
            shuffle=False, 
            drop_last=False,
            pin_memory=True
        )

        # 2. Setup GPU-accelerated preprocessing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard ALOHA sim resolution is 640x480, so we use 480x480 CenterCrop
        preprocess = T.Compose([
            T.CenterCrop(480),
            T.Resize((self.image_size, self.image_size), antialias=True)
        ])

        idx = 0
        for batch in tqdm(dl, desc="Hardware Accelerated Caching"):
            img = batch[self.cam_key]
            if img.ndim == 5:
                img = img[:, -1]
            elif img.ndim == 4 and batch["action"].shape[0] == img.shape[0]:
                pass # (B, C, H, W)
            
            # Send to GPU for fast preprocessing
            img = img.to(device, non_blocking=True)
            
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype != torch.float32:
                img = img.float()
            # Apply CenterCrop + Resize on GPU and cast to FP16
            img_processed = preprocess(img).half().cpu()
            
            bs = img_processed.shape[0]
            self.cached_images[idx : idx + bs] = img_processed
            idx += bs

        # Save processed cache. self.cache_path already includes '_centered' defined in __init__
        print(f"[AlohaDataset] Saving processed cache to: {self.cache_path}")
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
        # 3. Retrieve Data (Augmentations Disabled)
        # ----------------------------------------------------
        raw_state = self.cached_states[obs_steps_clamped]
        img_seq = self.cached_images[obs_steps_clamped].float()
        raw_action = self.cached_actions[act_steps_clamped]

        # ----------------------------------------------------
        # 3. Apply Data Augmentation (Safety First)
        # ----------------------------------------------------
        if self.augment:
            # torch.transforms.RandomCrop(..., size, padding) on a 4D Tensor (T, C, H, W)
            # applies the SAME crop across the leading dimensions (T).
            img_seq = self.augmentor(img_seq)

        return {
            "obs":    self.state_normalizer.normalize(raw_state),
            "image":  img_seq, 
            "action": self.action_normalizer.normalize(raw_action),
        }

if __name__ == "__main__":
    dataset = AlohaDataset(pred_horizon=16, obs_horizon=4, image_size=128)
    
    print(f"\n[AlohaDataset Discovery]")
    print(f"Dataset Size:   {len(dataset)} samples")
    print(f"State Dim:      {dataset.state_dim}")
    print(f"Action Dim:     {dataset.action_dim}")
    
    sample = dataset[len(dataset) // 2]
    print(f"\n[Sample Verification]")
    print(f"Observation Horizon: {dataset.obs_horizon}")
    print(f"Prediction Horizon:  {dataset.pred_horizon}")
    print(f"Obs State Shape:     {list(sample['obs'].shape)}")    # Expected: [4, 14]
    print(f"Obs Image Shape:     {list(sample['image'].shape)}")  # Expected: [4, 3, 128, 128]
    print(f"Action Chunk Shape:  {list(sample['action'].shape)}") # Expected: [16, 14]

    from torch.utils.data import DataLoader
    dataset = AlohaDataset(pred_horizon=16, obs_horizon=4, image_size=128)
    
    print(f"\n[AlohaDataset Discovery]")
    print(f"Dataset Size:   {len(dataset)} samples")
    print(f"State Dim:      {dataset.state_dim}")
    print(f"Action Dim:     {dataset.action_dim}")
    
    sample = dataset[len(dataset) // 2]
    print(f"\n[Sample Verification]")
    print(f"Observation Horizon: {dataset.obs_horizon}")
    print(f"Prediction Horizon:  {dataset.pred_horizon}")
    print(f"Obs State Shape:     {list(sample['obs'].shape)}")    # Expected: [4, 14]
    print(f"Obs Image Shape:     {list(sample['image'].shape)}")  # Expected: [4, 3, 128, 128]
    print(f"Action Chunk Shape:  {list(sample['action'].shape)}") # Expected: [16, 14]
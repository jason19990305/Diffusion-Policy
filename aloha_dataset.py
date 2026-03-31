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
    def __init__(self, min_val, max_val):
        self.min = min_val.copy().astype(np.float32)
        self.max = max_val.copy().astype(np.float32)
        self.range = np.where((self.max - self.min) == 0, 1e-5, self.max - self.min)

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def unnormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min

class AlohaDataset(Dataset):
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

        # 1. 快速載入 Dataset 結構
        self.lerobot_dataset = LeRobotDataset(self.DATASET_ID, video_backend="pyav")
        self.camera_keys = [k for k in self.lerobot_dataset.features if k.startswith("observation.images.")]
        
        # --- 優化 A: 批量獲取 States 與 Actions (不再使用 for loop) ---
        print("[AlohaDataset] Fast Caching States & Actions...")
        # 直接存取底層的 HuggingFace Dataset 物件，速度提升 100 倍
        hf_data = self.lerobot_dataset.hf_dataset
        self.cached_states = torch.from_numpy(np.array(hf_data["observation.state"])).float()
        self.cached_actions = torch.from_numpy(np.array(hf_data["action"])).float()
        
        # --- 處理步數索引 ---
        # 建立一個索引表，方便快速切換時間序列
        self.episode_data_index = self.lerobot_dataset.episode_data_index
        
        # --- 正規化統計 ---
        stats = self.lerobot_dataset.meta.stats
        self.state_normalizer  = Normalization(np.array(stats["observation.state"]["min"]), np.array(stats["observation.state"]["max"]))
        self.action_normalizer = Normalization(np.array(stats["action"]["min"]), np.array(stats["action"]["max"]))

        # --- 優化 B: 影像快取 (Memory Mapping) ---
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"aloha_img_obs{obs_horizon}_s{image_size}.pt")

        if os.path.isfile(self.cache_path):
            print(f"[AlohaDataset] Loading cache via mmap: {self.cache_path}")
            # 使用 mmap=True 實現瞬間載入，不佔用預先物理內存
            self.cached_images = torch.load(self.cache_path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._create_image_cache()

    def _create_image_cache(self):
        """ 一次性解碼影像並儲存 """
        print("[AlohaDataset] No cache found. Creating image cache (First time only)...")
        n = len(self.lerobot_dataset)
        num_cams = len(self.camera_keys)
        
        # 建立 Resize Transform
        resize = T.Resize((self.image_size, self.image_size), antialias=True)
        
        # 先分配空間 (使用 float16 節省空間)
        self.cached_images = torch.zeros((n, num_cams, 3, self.image_size, self.image_size), dtype=torch.float16)

        # 這裡仍然需要一次性的慢速解碼，但我們只針對「每一幀」解碼一次，而不是「每個 horizon」重複解碼
        for i in tqdm(range(n), desc="Decoding Videos"):
            item = self.lerobot_dataset[i]
            for c_idx, k in enumerate(self.camera_keys):
                # 拿取最後一幀影像 (T=1)
                img = item[k][-1] # (3, H, W)
                if img.dtype != torch.float32: img = img.float()
                if img.max() > 1.5: img /= 255.0
                self.cached_images[i, c_idx] = resize(img).half()

        print(f"[AlohaDataset] Saving cache to: {self.cache_path}")
        torch.save(self.cached_images, self.cache_path)

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        # 獲取時間範圍
        # 為了簡化，這裡假設數據是連續的，實際中需處理 Episode 邊界
        start_obs = max(0, idx - self.obs_horizon + 1)
        end_obs = idx + 1
        
        # 1. 取得影像序列 (obs_horizon, cams, 3, H, W)
        # 直接從 Tensor 切片非常快
        img_seq = self.cached_images[start_obs:end_obs]
        
        # 如果剛好在 Episode 開始處，長度不足時補齊 (Padding)
        if img_seq.shape[0] < self.obs_horizon:
            pad = img_seq[0:1].repeat(self.obs_horizon - img_seq.shape[0], 1, 1, 1, 1)
            img_seq = torch.cat([pad, img_seq], dim=0)

        # 2. 取得 States (obs_horizon, 14)
        raw_state = self.cached_states[start_obs:end_obs]
        if raw_state.shape[0] < self.obs_horizon:
            pad = raw_state[0:1].repeat(self.obs_horizon - raw_state.shape[0], 1)
            raw_state = torch.cat([pad, raw_state], dim=0)

        # 3. 取得 Actions (pred_horizon, 14)
        end_act = min(len(self.cached_actions), idx + self.pred_horizon)
        raw_action = self.cached_actions[idx:end_act]
        if raw_action.shape[0] < self.pred_horizon:
            pad = raw_action[-1:].repeat(self.pred_horizon - raw_action.shape[0], 1)
            raw_action = torch.cat([raw_action, pad], dim=0)

        # 正規化
        state_norm  = self.state_normalizer.normalize(raw_state.numpy())
        action_norm = self.action_normalizer.normalize(raw_action.numpy())

        return {
            "obs":    torch.from_numpy(state_norm).float(),
            "image":  img_seq.float(), 
            "action": torch.from_numpy(action_norm).float(),
        }
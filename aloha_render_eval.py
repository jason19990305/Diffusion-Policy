import os
import time
import collections
import argparse
import gc
import cv2
import torch
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
import torchvision.transforms as T

# Environment imports
import gym_aloha
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv

from noise_predictor import DiffusionPolicy
from aloha_dataset import AlohaDataset


class TensorTemporalEnsembling:
    """
    Temporal Ensembling for action smoothing across overlapping prediction horizons.
    """
    def __init__(self, pred_horizon, action_dim):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_sum = torch.zeros((pred_horizon, action_dim))
        self.action_count = torch.zeros((pred_horizon, 1))

    def update(self, predicted_action_seq):
        self.action_sum += predicted_action_seq
        self.action_count += 1

    def get_and_shift_actions(self, n_actions):
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


def parse_args():
    parser = argparse.ArgumentParser(description="ALOHA Evaluation (Spatial Softmax)")
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/aloha_diffusion_step_10000.pth")
    parser.add_argument("--output",       type=str, default="eval_aloha.mp4")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--fps",          type=int, default=50)
    parser.add_argument("--ddim_steps",   type=int, default=20)
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--obs_horizon",  type=int, default=4)
    parser.add_argument("--image_size",   type=int, default=224) # High-res support
    parser.add_argument("--execute_steps",type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_render] Running on: {DEVICE}")

    # ==========================================
    # 1. Dataset & Normalization Setup
    # ==========================================
    dataset = AlohaDataset(
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        image_size=args.image_size,
    )

    # ==========================================
    # 2. Environment Setup
    # ==========================================
    print(f"[aloha_render] Initializing ALOHA environment...")
    env_cfg = AlohaEnv(task="AlohaTransferCube-v0", render_mode="rgb_array", fps=args.fps)
    env_cfg.episode_length = 800
    envs = make_env(env_cfg)
    env = envs["aloha"][0]
    
    # ==========================================
    # 3. Model Definition (Spatial Softmax Architecture)
    # ==========================================
    model = DiffusionPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        embed_dim=512,
        num_heads=8,
        num_blocks=12,
        use_image=True,
        image_size=args.image_size, # No patch_size needed
        use_checkpoint=False
    ).to(DEVICE)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"[aloha_render] Loading model from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # ==========================================
    # 4. Inference Components Setup
    # ==========================================
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(args.ddim_steps)

    # CenterCrop to square and Resize to target (e.g., 480x480)
    resize_transform = T.Compose([
        T.CenterCrop(480),
        T.Resize((args.image_size, args.image_size), antialias=True)
    ])

    def prepare_obs(obs_dict, cam_key):
        """Processes raw observation dict into normalized Tensors."""
        # 1. State processing
        state = torch.as_tensor(obs_dict["agent_pos"][0], dtype=torch.float32)
        
        # 2. Image processing
        cam_name = cam_key.split(".")[-1]
        raw_img = obs_dict["pixels"][cam_name][0]
        img_tensor = torch.as_tensor(raw_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = resize_transform(img_tensor)
        
        return state, img_tensor

    # ==========================================
    # 5. Evaluation Loop
    # ==========================================
    base_output, ext = os.path.splitext(args.output)

    for ep in range(args.num_episodes):
        gc.collect()
        torch.cuda.empty_cache()

        out_path = f"{base_output}_{ep}{ext}" if args.num_episodes > 1 else args.output
        print(f"\n[aloha_render] Episode {ep+1}/{args.num_episodes}")

        obs, _ = env.reset()
        state, img_tensor = prepare_obs(obs, dataset.cam_key)

        # Buffers for history
        state_buffer = collections.deque([state] * args.obs_horizon, maxlen=args.obs_horizon)
        image_buffer = collections.deque([img_tensor] * args.obs_horizon, maxlen=args.obs_horizon)

        # Effective horizon for ensembling: pred_horizon - offset from history
        eff_horizon = args.pred_horizon - args.obs_horizon + 1
        ensembler = TensorTemporalEnsembling(eff_horizon, dataset.action_dim)

        frames = []
        max_steps, current_step = 800, 0
        pbar = tqdm(total=max_steps, desc=f"Evaluating")

        while current_step < max_steps:
            # Prepare Input Tensors
            stacked_states = torch.stack(list(state_buffer))
            obs_tensor = dataset.state_normalizer.normalize(stacked_states).unsqueeze(0).to(DEVICE)
            imgs_tensor = torch.stack(list(image_buffer)).unsqueeze(0).to(DEVICE)

            # DDIM Inference Loop
            noised_actions = torch.randn((1, args.pred_horizon, dataset.action_dim), device=DEVICE)
            with torch.no_grad():
                # Use BF16 for faster inference if supported
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for t in scheduler.timesteps:
                        noise_pred = model(
                            observations=obs_tensor,
                            diffusion_steps=torch.full((1,), t, device=DEVICE, dtype=torch.long),
                            noised_actions=noised_actions,
                            images=imgs_tensor
                        )
                        noised_actions = scheduler.step(noise_pred, t, noised_actions).prev_sample
            
            # Post-process actions
            action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu())
            future_actions = action_seq[args.obs_horizon - 1:]
            
            # Update temporal ensemble and get smoothed actions
            ensembler.update(future_actions)
            smoothed_actions = ensembler.get_and_shift_actions(args.execute_steps)
            
            # Execution Phase (Receding Horizon)
            for i in range(args.execute_steps):
                if current_step >= max_steps: break
                
                action = smoothed_actions[i]
                obs, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                
                # Update observation buffers
                state, img_tensor = prepare_obs(obs, dataset.cam_key)
                state_buffer.append(state)
                image_buffer.append(img_tensor)
                
                # Render frame
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, (list, tuple)): frame = frame[0]
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                current_step += 1
                pbar.update(1)

                if terminated.any():
                    print(f"\n[aloha_render] Task Success in {current_step} steps.")
                    current_step = max_steps 
                    break

        pbar.close()

        # Save Episode Video
        if frames:
            h, w, _ = frames[0].shape
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            for f in frames: writer.write(f)
            writer.release()
            print(f"[aloha_render] Video saved: {out_path}")

    env.close()
    print("\n[aloha_render] Evaluation finished.")

if __name__ == "__main__":
    main()
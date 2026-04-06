"""
aloha_render_eval.py
====================
Runs the trained ALOHA Diffusion Policy in the MuJoCo simulator and saves a video.
Uses receding horizon (predict 16, execute 8) for smooth closed-loop control.

Usage:
    python aloha_render_eval.py --checkpoint checkpoints/aloha_diffusion_step_6000.pth
"""

import os
import torch
import numpy as np
import cv2
import collections
import argparse
import gc
from tqdm import tqdm
from diffusers import DDIMScheduler
import torchvision.transforms as T

# lerobot v0.5 API
import gym_aloha
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv

from noise_predictor import DiffusionPolicy
from aloha_dataset import AlohaDataset


class TensorTemporalEnsembling:
    """
    Temporal Ensembling for PyTorch Tensors.
    Averages overlapping parts of multiple predictions to make the generated actions smoother.
    """
    def __init__(self, pred_horizon, action_dim):
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_sum = torch.zeros((pred_horizon, action_dim))
        self.action_count = torch.zeros((pred_horizon, 1))

    def update(self, predicted_action_seq):
        """Add the currently predicted sequence to the buffer."""
        self.action_sum += predicted_action_seq
        self.action_count += 1

    def get_and_shift_actions(self, n_actions):
        """Calculate the averaged actions and shift the buffer forward."""
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
    parser = argparse.ArgumentParser(description="Render Evaluation for ALOHA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aloha_diffusion_step_RTX5090_10000.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="eval_aloha.mp4",
                        help="Output video filename (will be indexed if num_episodes > 1)")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--fps", type=int, default=50, help="Simulation FPS")
    parser.add_argument("--ddim_steps", type=int, default=20, help="DDIM inference steps")
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--execute_steps", type=int, default=8, help="Steps to execute per chunk")
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_render] Device: {DEVICE}")

    # 1. Load Dataset for Normalizers
    # We use the dataset's normalizers to ensure consistency with training
    dataset = AlohaDataset(
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        image_size=args.image_size,
    )

    # 2. Setup environment
    print(f"[aloha_render] Initializing ALOHA environment...")
    env_cfg = AlohaEnv(task="AlohaTransferCube-v0", render_mode="rgb_array", fps=args.fps)
    envs = make_env(env_cfg)
    env = envs["aloha"][0] # Get the sync vector env
    
    # 3. Load Model
    model = DiffusionPolicy(
        action_dim  = dataset.action_dim,
        state_dim   = dataset.state_dim,
        embed_dim   = 512,
        num_heads   = 8,
        num_blocks  = 12,
        use_image   = True,
        image_size  = args.image_size,
        patch_size  = args.patch_size,
        use_checkpoint = False # Not needed for eval
    ).to(DEVICE)

    if not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint {args.checkpoint} not found. Trying latest...")
        # fallback to aloha_diffusion.pth if step-based name is missing
        args.checkpoint = "checkpoints/aloha_diffusion_H100.pth"
    print(f"[aloha_render] Loading model from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"[aloha_render] Model loaded from {args.checkpoint}")

    # 4. Setup Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps = 100,
        beta_schedule       = "squaredcos_cap_v2",
        clip_sample         = True,
        set_alpha_to_one    = True,
        prediction_type     = "epsilon",
    )
    scheduler.set_timesteps(args.ddim_steps)

    # 5. Image Transformation
    # CenterCrop(480) is used to maintain 1:1 aspect ratio for ALOHA's 640x480 images
    resize_transform = T.Compose([
        T.CenterCrop(480),
        T.Resize((args.image_size, args.image_size), antialias=True)
    ])


    # ---------------------------------------------------------------- #
    # 5. Observation Handling (Updated for Tensors)
    # ---------------------------------------------------------------- #
    def get_obs_data(obs_dict, cam_keys):
        # 1. State: (1, 14) -> (14,)
        s_val = obs_dict["agent_pos"][0]
        # Keep as Tensor instead of converting to numpy
        state = torch.as_tensor(s_val, dtype=torch.float32).cpu()
        
        # 2. Dynamic Images
        cam_name = cam_keys.split(".")[-1]
        raw = obs_dict["pixels"][cam_name][0]
        
        # Keep as Tensor
        img_tensor = torch.as_tensor(raw, dtype=torch.float32).cpu()
        # (H, W, 3) -> (3, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)
        # [0,255] -> [0,1]
        img_tensor /= 255.0
        # apply composed transform: CenterCrop(480) + Resize(image_size)
        img_tensor = resize_transform(img_tensor)
        
        return state, img_tensor

    # ---------------------------------------------------------------- #
    # 6. Multi-Episode Loop                                            #
    # ---------------------------------------------------------------- #
    base_output, ext = os.path.splitext(args.output)

    for ep in range(args.num_episodes):
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

        out_path = f"{base_output}_{ep}{ext}" if args.num_episodes > 1 else args.output
        print(f"\n[aloha_render] Starting Episode {ep+1}/{args.num_episodes} ...")

        obs, _ = env.reset()
        state, img_tensor = get_obs_data(obs, dataset.cam_key)

        # Buffers to store history (obs_horizon)
        state_buffer = collections.deque([state] * args.obs_horizon, maxlen=args.obs_horizon)
        image_buffer = collections.deque([img_tensor] * args.obs_horizon, maxlen=args.obs_horizon)

        # Initialize Temporal Ensembling
        # The usable future actions start at index (obs_horizon - 1), 
        # so the effective future horizon is pred_horizon - obs_horizon + 1
        effective_horizon = args.pred_horizon - args.obs_horizon + 1
        temporal_ensembler = TensorTemporalEnsembling(
            pred_horizon=effective_horizon, 
            action_dim=dataset.action_dim
        )

        frames =[] # To save video
        max_steps, current_step = 800, 0
        pbar = tqdm(total=max_steps, desc=f"Ep {ep}")

        while current_step < max_steps:
            # ----------------------------------------------------
            # Prepare Tensors (Updated for pure PyTorch normalizer)
            # ----------------------------------------------------
            
            # Normalize State: (1, obs_horizon, 14)
            stacked_states = torch.stack(list(state_buffer))
            obs_tensor = dataset.state_normalizer.normalize(stacked_states).unsqueeze(0).to(DEVICE)
            
            # Images: (1, obs_horizon, 3, image_size, image_size)
            imgs_tensor = torch.stack(list(image_buffer)).unsqueeze(0).to(DEVICE)

            # DDIM Inference
            noised_actions = torch.randn((1, args.pred_horizon, dataset.action_dim), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise = model(
                        observations=obs_tensor,
                        diffusion_steps=torch.full((1,), t, device=DEVICE, dtype=torch.long),
                        noised_actions=noised_actions,
                        images=imgs_tensor
                    )
                    noised_actions = scheduler.step(noise, t, noised_actions).prev_sample
            
            # Unnormalize actions directly via TensorNormalizer
            noised_actions_cpu = noised_actions.squeeze(0).cpu()
            action_seq = dataset.action_normalizer.unnormalize(noised_actions_cpu)
            
            start_idx = args.obs_horizon - 1
            future_actions = action_seq[start_idx:]  # slice only the future predictions (shape: effective_horizon x action_dim)
            
            # temporal ensembling update
            temporal_ensembler.update(future_actions)
            smoothed_actions = temporal_ensembler.get_and_shift_actions(args.execute_steps)
            
            # Execute Receding Horizon steps
            for i in range(args.execute_steps):
                if current_step >= max_steps: break
                
                # action is a PyTorch tensor, MuJoCo env can consume it properly
                action = smoothed_actions[i]
                obs, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                
                state, img_tensor = get_obs_data(obs, dataset.cam_key)
                state_buffer.append(state)
                image_buffer.append(img_tensor)
                
                try:
                    frame = env.render() 
                    if frame is not None:
                        
                        if isinstance(frame, (list, tuple)): 
                            frame = frame[0]
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Render warning: {e}")

                
                current_step += 1
                pbar.update(1)

                if terminated.any():
                    print(f"\n[aloha_render] Episode {ep} finished early (Success).")
                    current_step = max_steps 
                    break

        pbar.close()

        # Save Video for this episode
        if len(frames) > 0:
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"[aloha_render] Video saved to: {out_path}")
        else:
            print(f"[aloha_render] Error: No frames collected for episode {ep}.")

    env.close()
    print("\n[aloha_render] All episodes completed.")


if __name__ == "__main__":
    main()
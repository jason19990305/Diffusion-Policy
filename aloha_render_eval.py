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
from tqdm import tqdm
from diffusers import DDIMScheduler
import torchvision.transforms as T

# lerobot v0.5 API
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv

from noise_predictor import DiffusionPolicy
from aloha_dataset import AlohaDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Render Evaluation for ALOHA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/aloha_diffusion.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="eval_aloha.mp4",
                        help="Output video filename")
    parser.add_argument("--fps", type=int, default=50, help="Simulation FPS")
    parser.add_argument("--ddim_steps", type=int, default=10, help="DDIM inference steps")
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=4)
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
        prefetch_images=False # Not needed for eval
    )

    # 2. Setup environment
    print(f"[aloha_render] Initializing ALOHA environment...")
    env_cfg = AlohaEnv(task="AlohaInsertion-v0", render_mode="rgb_array", fps=args.fps)
    envs = make_env(env_cfg)
    env = envs["aloha"][0] # Get the sync vector env
    
    # 3. Load Model
    model = DiffusionPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        embed_dim=512,
        num_heads=8,
        num_blocks=12,
        use_image=True,
        image_size=96,
        patch_size=16
    ).to(DEVICE)

    if not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint {args.checkpoint} not found. Trying latest...")
        # fallback to aloha_diffusion.pth if step-based name is missing
        args.checkpoint = "checkpoints/aloha_diffusion.pth"

    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"[aloha_render] Model loaded from {args.checkpoint}")

    # 4. Setup Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(args.ddim_steps)

    # 5. Image Transformation (Match training: resize to 96x96)
    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((96, 96)),
        T.ToTensor(),
    ])

    # ---------------------------------------------------------------- #
    # Execution Loop                                                     #
    # ---------------------------------------------------------------- #
    obs, _ = env.reset()
    
    # ---------------------------------------------------------------- #
    # 5. Observation Handling
    # ---------------------------------------------------------------- #
    def get_obs_data(obs_dict):
        # State: (1, 14) -> (14,)
        s_val = obs_dict["agent_pos"][0]
        state = s_val.cpu().numpy() if torch.is_tensor(s_val) else s_val
        
        # Image: (1, H, W, 3) -> (H, W, 3) -> (3, 96, 96)
        i_val = obs_dict["pixels"]["top"][0]
        img_raw = i_val.cpu().numpy() if torch.is_tensor(i_val) else i_val
        
        return state, img_transform(img_raw)

    obs, _ = env.reset()
    state, img_tensor = get_obs_data(obs)

    # Buffers to store history (obs_horizon = 2)
    state_buffer = collections.deque([state] * args.obs_horizon, maxlen=args.obs_horizon)
    image_buffer = collections.deque([img_tensor] * args.obs_horizon, maxlen=args.obs_horizon)

    frames = [] # To save video
    max_steps, current_step = 400, 0
    pbar = tqdm(total=max_steps, desc="Simulating ALOHA")

    while current_step < max_steps:
        # Prepare Tensors
        # Normalized State: (1, obs_horizon, 14)
        n_obs = dataset.state_normalizer.normalize(np.stack(state_buffer))
        obs_tensor = torch.from_numpy(n_obs).float().unsqueeze(0).to(DEVICE)
        
        # Images: (1, obs_horizon, 3, 96, 96)
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
        
        # Unnormalize actions: (16, 14)
        action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())

        # Execute Receding Horizon steps
        for i in range(args.execute_steps):
            if current_step >= max_steps: break
            
            action = action_seq[i]
            obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).unsqueeze(0))
            
            # Update buffers with new observation
            state, img_tensor = get_obs_data(obs)
            state_buffer.append(state)
            image_buffer.append(img_tensor)
            
            # Record frame (High res for video)
            # Depending on gym_aloha, render() might be needed or images are in obs
            # If render() is rgb_array, it returns the frame.
            frame = env.render() 
            if frame is not None:
                # If vectorized env returns a list of frames, get the first one
                if isinstance(frame, (list, tuple)): frame = frame[0]
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            current_step += 1
            pbar.update(1)

            if terminated.any() or truncated.any():
                print("\n[aloha_render] Episode finished early.")
                current_step = max_steps # break outer loop
                break

    env.close()
    pbar.close()

    # 6. Save Video
    if len(frames) > 0:
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        print(f"\n[aloha_render] Video saved to: {args.output}")
    else:
        print("\n[aloha_render] Error: No frames collected.")


if __name__ == "__main__":
    main()

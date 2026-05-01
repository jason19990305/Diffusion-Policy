import os
import time
import collections
import argparse
import gc
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

# Environment imports
import gym_aloha
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv

from utils.noise_predictor import DiffusionPolicy
from utils.consistency import ConsistencyPolicy, ConsistencySampler
from aloha_dataset import AlohaDataset

from utils.ensembling import TensorTemporalEnsembler

def parse_args():
    parser = argparse.ArgumentParser(description="ALOHA Consistency Policy Evaluation")
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/aloha_consistency_policy.pth")
    parser.add_argument("--output",       type=str, default="assets/aloha_consistency_eval.mp4")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--fps",          type=int, default=50)
    parser.add_argument("--consistency_steps", type=int, default=2) # 2-step fast inference
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--obs_horizon",  type=int, default=4)
    parser.add_argument("--image_size",   type=int, default=224) 
    parser.add_argument("--execute_steps",type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_consistency_eval] Running on: {DEVICE}")

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
    print(f"[aloha_consistency_eval] Initializing ALOHA environment...")
    env_cfg = AlohaEnv(task="AlohaTransferCube-v0", render_mode="rgb_array", fps=args.fps)
    env_cfg.episode_length = 800
    envs = make_env(env_cfg)
    env = envs["aloha"][0]
    
    # ==========================================
    # 3. Model Definition
    # ==========================================
    inner_model = DiffusionPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        embed_dim=512,
        num_heads=8,
        num_blocks=12,
        use_image=True,
        image_size=args.image_size, 
        use_checkpoint=False
    ).to(DEVICE)

    model = ConsistencyPolicy(inner_model).to(DEVICE)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"[aloha_consistency_eval] Loading model from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    sampler = ConsistencySampler(model)

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
    # 4. Evaluation Loop
    # ==========================================
    base_output, ext = os.path.splitext(args.output)
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for ep in range(args.num_episodes):
        gc.collect()
        torch.cuda.empty_cache()

        out_path = f"{base_output}_{ep}{ext}" if args.num_episodes > 1 else args.output
        print(f"\n[aloha_consistency_eval] Episode {ep+1}/{args.num_episodes}")

        obs, _ = env.reset()
        state, img_tensor = prepare_obs(obs, dataset.cam_key)

        # Buffers for history
        state_buffer = collections.deque([state] * args.obs_horizon, maxlen=args.obs_horizon)
        image_buffer = collections.deque([img_tensor] * args.obs_horizon, maxlen=args.obs_horizon)

        # Effective horizon for ensembling: pred_horizon - offset from history
        eff_horizon = args.pred_horizon - args.obs_horizon + 1
        ensembler = TensorTemporalEnsembler(eff_horizon, dataset.action_dim)

        frames = []
        max_steps, current_step = 800, 0
        pbar = tqdm(total=max_steps, desc=f"Evaluating")

        while current_step < max_steps:
            # Prepare Input Tensors
            stacked_states = torch.stack(list(state_buffer))
            obs_tensor = dataset.state_normalizer.normalize(stacked_states).unsqueeze(0).to(DEVICE)
            imgs_tensor = torch.stack(list(image_buffer)).unsqueeze(0).to(DEVICE)

            # 2-Step Consistency Sampling
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    best_action_seq = sampler.sample(
                        observations=obs_tensor,
                        pred_horizon=args.pred_horizon,
                        action_dim=dataset.action_dim,
                        images=imgs_tensor,
                        steps=args.consistency_steps,
                        sigma_start=1.0
                    )
            
            # Post-process actions
            action_seq = dataset.action_normalizer.unnormalize(best_action_seq.squeeze(0).cpu())
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
                    print(f"\n[aloha_consistency_eval] Task Success in {current_step} steps.")
                    current_step = max_steps 
                    break

        pbar.close()

        # Save Episode Video
        if frames:
            h, w, _ = frames[0].shape
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
            for f in frames: writer.write(f)
            writer.release()
            print(f"[aloha_consistency_eval] Video saved: {out_path}")

    env.close()
    print("\n[aloha_consistency_eval] Evaluation finished.")

if __name__ == "__main__":
    main()

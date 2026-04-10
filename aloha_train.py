import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from noise_predictor import DiffusionPolicy, EMA
from diffusers import DDIMScheduler
from aloha_dataset import AlohaDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ALOHA Diffusion Policy Training")
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--total_steps",   type=int,   default=3e5)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--num_workers",   type=int,   default=0) 
    parser.add_argument("--save_interval", type=int,   default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    # Enable TF32 for faster matmuls on Ampere/Ada GPUs
    torch.set_float32_matmul_precision('high')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_train] Using device: {DEVICE}")

    args = parse_args()

    # Diffusion & Model architecture settings
    TIMESTEPS    = 100        # Total diffusion steps (T)
    PRED_HORIZON = 32         # Action prediction horizon (1.28s @ 50fps)
    OBS_HORIZON  = 4          # Observation history length
    EMBED_DIM    = 512        # Transformer embedding dim
    NUM_HEADS    = 8          # Attention heads
    MLP_RATIO    = 4.0        # SwiGLU hidden-dim ratio
    DEPTH        = 12         # Transformer blocks

    # Image settings
    IMAGE_SIZE   = 224           
    IN_CHANNELS  = 3             

    # Training loop settings
    WARMUP_STEPS = 3000
    LOG_INTERVAL = 100         
    SAVE_DIR     = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 1. Dataset & DataLoader
    # ==========================================
    dataset = AlohaDataset(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        image_size=IMAGE_SIZE,
        augment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"[aloha_train] Dataset size: {len(dataset)} | Batches/epoch: {len(dataloader)}")

    # ==========================================
    # 2. Model & EMA Definition
    # ==========================================
    model = DiffusionPolicy(
        action_dim=dataset.action_dim, 
        state_dim=dataset.state_dim,    
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_blocks=DEPTH,
        use_image=True,
        in_channels=IN_CHANNELS,
        image_size=IMAGE_SIZE,
        use_checkpoint=True,  # Enable gradient checkpointing to save VRAM
    ).to(DEVICE)

    # Calculate token sequence length for logging
    total_tokens = OBS_HORIZON + OBS_HORIZON + 1 + PRED_HORIZON

    print(f"[aloha_train] Transformer sequence length: {total_tokens} tokens")

    model.train()
    ema_model = EMA(model, beta=0.995)

    # ==========================================
    # 3. Optimizer, Scheduler & Diffusion Setup
    # ==========================================
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse_loss  = nn.MSELoss()
    
    # Cosine Annealing with Linear Warmup
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        # Cosine decay after warmup
        progress = float(current_step - WARMUP_STEPS) / float(max(1, args.total_steps - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"[aloha_train] Starting training for {args.total_steps} steps ...")
    global_step = 0
    t_start = time.time()
    
    pbar = tqdm(total=args.total_steps, desc="ALOHA Training")

    while global_step < args.total_steps:
        for batch in dataloader:
            if global_step >= args.total_steps:
                break
            
            # --- Prepare Data ---
            obs     = batch["obs"].to(DEVICE, non_blocking=True)
            images  = batch["image"].to(DEVICE, non_blocking=True)
            actions = batch["action"].to(DEVICE, non_blocking=True)

            # --- Forward Diffusion ---
            # Sample random timesteps
            k = torch.randint(0, TIMESTEPS, (obs.shape[0],), device=DEVICE)
            # Add noise to ground-truth actions
            noise = torch.randn_like(actions)
            noised_actions = noise_scheduler.add_noise(actions, noise, k)

            # --- Forward Pass (BF16 Autocast) ---
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_noise = model(
                    observations=obs,
                    diffusion_steps=k,
                    noised_actions=noised_actions,
                    images=images,
                )
                loss = mse_loss(predicted_noise, noise)

            # --- Backward Pass & Optimize ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent divergence
            
            optimizer.step()
            scheduler.step()
            ema_model.update(model)

            # --- Logging & Checkpointing ---
            global_step += 1
            pbar.update(1)

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                it_per_sec = global_step / max(elapsed, 1e-6)
                eta_h = (args.total_steps - global_step) / max(it_per_sec, 1e-6) / 3600
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.5f}", 
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    "it/s": f"{it_per_sec:.2f}",
                    "ETA": f"{eta_h:.1f}h"
                })

            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"aloha_diffusion_step_{global_step}.pth")
                ema_model.save_pretrained(ckpt_path)
                ema_model.save_pretrained(os.path.join(SAVE_DIR, "aloha_diffusion.pth"))
                # Print nicely without breaking tqdm formatting
                tqdm.write(f"[aloha_train] Checkpoint saved at step {global_step}")

    pbar.close()
    
    # Save final model
    final_path = os.path.join(SAVE_DIR, "aloha_diffusion.pth")
    ema_model.save_pretrained(final_path)
    print(f"[aloha_train] Training complete. Final model -> {final_path}")
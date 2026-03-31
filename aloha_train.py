import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import argparse
import numpy as np

from noise_predictor import DiffusionPolicy, EMA
from diffusers import DDIMScheduler
from aloha_dataset import AlohaDataset


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 0. GPU Speed Optimizations                                           #
    # ------------------------------------------------------------------ #
    # Enable TF32 for much faster matmuls on RTX 30/40/50 series GPUs
    torch.set_float32_matmul_precision('high')

    # ------------------------------------------------------------------ #
    # 1. Hyperparameters                                                   #
    # ------------------------------------------------------------------ #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_train] Using device: {DEVICE}")

    # Diffusion settings
    TIMESTEPS    = 100          # Total diffusion training steps (T)
    PRED_HORIZON = 16           # Action prediction horizon
    OBS_HORIZON  = 4            # Observation (state + image) history length

    # Model architecture
    EMBED_DIM  = 512            # Transformer embedding dimension
    NUM_HEADS  = 8              # Multi-head attention heads
    MLP_RATIO  = 4.0            # SwiGLU hidden-dim ratio
    DEPTH      = 12             # Number of Transformer blocks

    # Image settings (matched to AlohaDataset)
    IMAGE_SIZE  = 128           
    PATCH_SIZE  = 8             # Each img: (128/8)^2 = 256 patches
    IN_CHANNELS = 3             

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=0) 

    args = parser.parse_args()

    # Training settings
    BATCH_SIZE   = args.batch_size          
    TOTAL_STEPS  = args.total_steps        
    LOG_INTERVAL  = 100         
    SAVE_INTERVAL = 5000        
    LR            = args.lr
    WARMUP_STEPS  = 1000        

    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. Dataset and DataLoader                                            #
    # ------------------------------------------------------------------ #
    dataset = AlohaDataset(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        image_size=IMAGE_SIZE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = args.num_workers,        # Set to 0 to eliminate WSL/multiprocessing lag
        pin_memory  = True,
        drop_last=True,
        persistent_workers=False,
    )
    print(f"[aloha_train] Dataset size: {len(dataset)} | "
          f"Batches/epoch: {len(dataloader)}")

    # ------------------------------------------------------------------ #
    # 3. Model, EMA                                                        #
    # ------------------------------------------------------------------ #
    model = DiffusionPolicy(
        action_dim  = dataset.action_dim,   # 14
        state_dim   = dataset.state_dim,    # 14
        embed_dim   = EMBED_DIM,
        num_heads   = NUM_HEADS,
        mlp_ratio   = MLP_RATIO,
        num_blocks  = DEPTH,
        # --- Image conditioning ---
        use_image   = True,
        in_channels = IN_CHANNELS,
        image_size  = IMAGE_SIZE,
        patch_size  = PATCH_SIZE,
        use_checkpoint = True,  # 啟用梯度檢查點以節省顯存
    ).to(DEVICE)

    # Token sequence per sample:
    #   image : obs_horizon * num_cameras * (image_size/patch_size)^2
    num_cameras = len(dataset.camera_keys)
    total_tokens = OBS_HORIZON * num_cameras * (IMAGE_SIZE // PATCH_SIZE) ** 2 + OBS_HORIZON + 1 + PRED_HORIZON
    print(f"[aloha_train] Transformer sequence length: {total_tokens} tokens")

    model.train()
    ema_model = EMA(model, beta=0.995)

    # ------------------------------------------------------------------ #
    # 4. Scheduler, Optimizer, AMP Scaler                                  #
    # ------------------------------------------------------------------ #
    noise_scheduler = DDIMScheduler(
        num_train_timesteps = TIMESTEPS,
        beta_schedule       = "squaredcos_cap_v2",
        clip_sample         = True,
        set_alpha_to_one    = True,
        prediction_type     = "epsilon",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Cosine Annealing with Linear Warmup
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        # After warmup, cosine decay to zero
        progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mse_loss  = nn.MSELoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=True)

    # ------------------------------------------------------------------ #
    # 5. Step-based Training Loop                                          #
    # ------------------------------------------------------------------ #
    print(f"[aloha_train] Starting training for {TOTAL_STEPS} steps ...")
    global_step = 0
    pbar = tqdm(total=TOTAL_STEPS, desc="ALOHA Training")
    t_start = time.time()

    while global_step < TOTAL_STEPS:
        for batch in dataloader:
            if global_step >= TOTAL_STEPS:
                break
            
            # 5.1 Prepare data
            obs    = batch["obs"].to(DEVICE, non_blocking=True)     # (B, obs_horizon, 14)
            images = batch["image"].to(DEVICE, non_blocking=True)   # (B, obs_horizon, 3, H, W)
            actions = batch["action"].to(DEVICE, non_blocking=True) # (B, pred_horizon, 14)

            # 5.1 Sample random diffusion timesteps
            k = torch.randint(0, TIMESTEPS, (obs.shape[0],), device=DEVICE)

            # 5.2 Add noise to ground-truth actions
            noise         = torch.randn_like(actions)
            noised_actions = noise_scheduler.add_noise(actions, noise, k)

            # 5.3 Forward pass with AMP (BF16 is faster on RTX 50 series)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_noise = model(
                    observations  = obs,
                    diffusion_steps = k,
                    noised_actions  = noised_actions,
                    images          = images,         # <-- image conditioning
                )
                loss = mse_loss(predicted_noise, noise)

            # 5.4 Backprop with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Update learning rate every step
            
            # 5.5 EMA update
            ema_model.update(model)

            global_step += 1
            pbar.update(1)

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                it_per_sec = global_step / max(elapsed, 1e-6)
                eta_h = (TOTAL_STEPS - global_step) / max(it_per_sec, 1e-6) / 3600
                pbar.set_postfix({
                    "Loss": f"{loss.item():.5f}", 
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    "it/s": f"{it_per_sec:.2f}",
                    "ETA": f"{eta_h:.1f}h"
                })

            if global_step % SAVE_INTERVAL == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"aloha_diffusion_step_{global_step}.pth")
                ema_model.save_pretrained(ckpt_path)
                ema_model.save_pretrained(os.path.join(SAVE_DIR, "aloha_diffusion.pth"))
                print(f"\n[aloha_train] Checkpoint saved at step {global_step}")

    pbar.close()
    ema_model.save_pretrained(os.path.join(SAVE_DIR, "aloha_diffusion.pth"))
    print(f"[aloha_train] Training complete. "
          f"Final model -> {os.path.join(SAVE_DIR, 'aloha_diffusion.pth')}")

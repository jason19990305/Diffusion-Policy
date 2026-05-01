import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from utils.noise_predictor import DiffusionPolicy, EMA
from utils.consistency import ConsistencyPolicy
from utils.consistency_loss import ConsistencyDistillationLoss
from diffusers import DDIMScheduler
from aloha_dataset import AlohaDataset

def parse_args():
    parser = argparse.ArgumentParser(description="ALOHA Consistency Policy Distillation")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--total_steps",   type=int,   default=100000) # 100k for distillation
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--num_workers",   type=int,   default=0) 
    parser.add_argument("--save_interval", type=int,   default=10000)
    return parser.parse_args()

if __name__ == "__main__":
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    torch.set_float32_matmul_precision('high')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[aloha_consistency_distill] Using device: {DEVICE}")

    args = parse_args()

    # Diffusion & Model architecture settings
    TIMESTEPS    = 100        
    PRED_HORIZON = 32         
    OBS_HORIZON  = 4          
    EMBED_DIM    = 512        
    NUM_HEADS    = 8          
    MLP_RATIO    = 4.0        
    DEPTH        = 12         

    # Image settings
    IMAGE_SIZE   = 224           
    IN_CHANNELS  = 3             

    # Training loop settings
    WARMUP_STEPS = 3000
    LOG_INTERVAL = 100         
    SAVE_DIR     = "checkpoints"
    CHECKPOINT_PATH = "checkpoints/aloha_diffusion_step_400000.pth"
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
    print(f"[aloha_consistency_distill] Dataset size: {len(dataset)} | Batches/epoch: {len(dataloader)}")

    # ==========================================
    # 2. Models Initialization
    # ==========================================
    # Initialize Teacher
    teacher_inner = DiffusionPolicy(
        action_dim=dataset.action_dim, 
        state_dim=dataset.state_dim,    
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_blocks=DEPTH,
        use_image=True,
        in_channels=IN_CHANNELS,
        image_size=IMAGE_SIZE,
        use_checkpoint=True,
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        teacher_inner.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        print(f"[aloha_consistency_distill] Loaded teacher from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found at {CHECKPOINT_PATH}")
        
    teacher_inner.eval()
    for param in teacher_inner.parameters():
        param.requires_grad = False

    # Initialize Student & Target
    student_inner = copy.deepcopy(teacher_inner)
    for param in student_inner.parameters():
        param.requires_grad = True
    student_model = ConsistencyPolicy(student_inner, sigma_data=0.1).to(DEVICE)
    student_model.train()
    
    target_inner = copy.deepcopy(teacher_inner)
    target_model = ConsistencyPolicy(target_inner, sigma_data=0.1).to(DEVICE)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False

    target_ema = EMA(student_model, beta=0.999)

    # ==========================================
    # 3. Optimizer, Scheduler & Diffusion Setup
    # ==========================================
    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(TIMESTEPS)
    
    loss_fn = ConsistencyDistillationLoss(
        teacher_model=teacher_inner,
        solver=scheduler
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-3, eps=1e-5)
    
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, args.total_steps - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"[aloha_consistency_distill] Starting training for {args.total_steps} steps ...")
    global_step = 0
    t_start = time.time()
    
    pbar = tqdm(total=args.total_steps, desc="Distilling")

    while global_step < args.total_steps:
        for batch in dataloader:
            if global_step >= args.total_steps:
                break
            
            # --- Prepare Data ---
            obs     = batch["obs"].to(DEVICE, non_blocking=True)
            images  = batch["image"].to(DEVICE, non_blocking=True)
            actions = batch["action"].to(DEVICE, non_blocking=True)

            # Sample t and t_next
            t_idx = torch.randint(0, TIMESTEPS - 1, (1,), device=DEVICE)
            t_indices = t_idx.repeat(obs.shape[0])
            t_next_indices = t_indices + 1

            # --- Forward Pass (BF16 Autocast) ---
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(
                    student_model=student_model,
                    target_model=target_model,
                    observations=obs,
                    actions=actions,
                    t=t_indices,
                    t_next=t_next_indices,
                    images=images # Ensure images are passed!
                )

            # --- Backward Pass & Optimize ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            target_ema.update(student_model)
            target_ema.copy_to(target_model)

            # --- Logging & Checkpointing ---
            global_step += 1
            pbar.update(1)

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                it_per_sec = global_step / max(elapsed, 1e-6)
                eta_h = (args.total_steps - global_step) / max(it_per_sec, 1e-6) / 3600
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.5f}", 
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.1e}",
                    "it/s": f"{it_per_sec:.2f}",
                    "ETA": f"{eta_h:.1f}h"
                })

            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(SAVE_DIR, f"aloha_consistency_policy_step_{global_step}.pth")
                torch.save(student_model.state_dict(), ckpt_path)
                torch.save(student_model.state_dict(), os.path.join(SAVE_DIR, "aloha_consistency_policy.pth"))
                tqdm.write(f"[aloha_consistency_distill] Checkpoint saved at step {global_step}")

    pbar.close()
    
    # Save final model
    final_path = os.path.join(SAVE_DIR, "aloha_consistency_policy.pth")
    torch.save(student_model.state_dict(), final_path)
    print(f"[aloha_consistency_distill] Training complete. Final model -> {final_path}")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from utils.noise_predictor import DiffusionPolicy, EMA
from utils.consistency import ConsistencyPolicy
from utils.consistency_loss import ConsistencyDistillationLoss
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

def train_consistency():
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[point_maze_consistency_distill] Using device: {DEVICE}")

    # Hyperparameters (Must match Teacher)
    TIMESTEPS = 100         
    EMBED_DIM = 256         
    NUM_HEADS = 8           
    MLP_RATIO = 4.0         
    PRED_HORIZON = 32       
    OBS_HORIZON = 2         
    DEPTH = 8               
    
    # Training Hyperparameters
    BATCH_SIZE = 1024       # Large batch size for good gradient estimation
    LR = 2e-4               # Consistency training LR
    TOTAL_STEPS = 10000     # Distillation takes fewer steps than training from scratch
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 2000
    CHECKPOINT_PATH = "checkpoints/point_maze_diffusion.pth"
    SAVE_DIR = "checkpoints"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ==========================================
    # 1. Dataset & DataLoader
    # ==========================================
    print("[point_maze_consistency_distill] Initializing Dataset...")
    dataset = PointMazeDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )

    # ==========================================
    # 2. Models Initialization
    # ==========================================
    print("[point_maze_consistency_distill] Loading Teacher Model...")
    # Initialize Teacher
    teacher_inner = DiffusionPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_blocks=DEPTH
    ).to(DEVICE)
    
    # Load teacher weights
    if os.path.exists(CHECKPOINT_PATH):
        teacher_inner.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        print(f"[point_maze_consistency_distill] Loaded teacher from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found at {CHECKPOINT_PATH}")
    
    teacher_inner.eval()
    for param in teacher_inner.parameters():
        param.requires_grad = False

    # Initialize Student & Target (Copy Teacher Weights)
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

    # Target model is updated via EMA of Student
    target_ema = EMA(student_model, beta=0.999)

    # ==========================================
    # 3. Solver & Loss
    # ==========================================
    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,   
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(TIMESTEPS) 
    
    loss_fn = ConsistencyDistillationLoss(
        teacher_model=teacher_inner,
        solver=scheduler
    )
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"[point_maze_consistency_distill] Starting distillation for {TOTAL_STEPS} steps...")
    
    global_step = 0
    pbar = tqdm(total=TOTAL_STEPS, desc="Distilling")
    
    while global_step < TOTAL_STEPS:
        for batch in dataloader:
            if global_step >= TOTAL_STEPS:
                break
                
            obs = batch['obs'].to(DEVICE, non_blocking=True)
            actions = batch['action'].to(DEVICE, non_blocking=True)
            
            # Sample t and t_next
            t_idx = torch.randint(0, TIMESTEPS - 1, (1,), device=DEVICE)
            t_indices = t_idx.repeat(obs.shape[0])
            t_next_indices = t_indices + 1
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = loss_fn(
                    student_model=student_model,
                    target_model=target_model,
                    observations=obs,
                    actions=actions,
                    t=t_indices,
                    t_next=t_next_indices
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            target_ema.update(student_model)
            target_ema.copy_to(target_model)
            
            global_step += 1
            pbar.update(1)
            
            if global_step % LOG_INTERVAL == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Step": global_step})
                
            if global_step % SAVE_INTERVAL == 0 or global_step == TOTAL_STEPS:
                checkpoint_path = os.path.join(SAVE_DIR, f"point_maze_consistency_policy_{global_step}.pth")
                torch.save(student_model.state_dict(), checkpoint_path)
                torch.save(student_model.state_dict(), os.path.join(SAVE_DIR, "point_maze_consistency_policy.pth"))
                tqdm.write(f"[point_maze_consistency_distill] Saved student model to {checkpoint_path}")

    pbar.close()
    print("[point_maze_consistency_distill] Distillation completed.")

if __name__ == "__main__":
    train_consistency()

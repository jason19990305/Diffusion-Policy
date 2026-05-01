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
from trajectory_dataset import TrajectoryDataset

def train_consistency():
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trajectory_distillation] Using device: {DEVICE}")

    # Hyperparameters (Must match Teacher)
    TIMESTEPS = 100         
    EMBED_DIM = 256         
    NUM_HEADS = 8           
    MLP_RATIO = 4.0         
    PRED_HORIZON = 16       
    OBS_HORIZON = 8         
    DEPTH = 4               
    
    # Training Hyperparameters
    BATCH_SIZE = 32         # Increased for better gradient estimation
    LR = 5e-4               # Increased to accelerate learning with teacher anchoring
    EPOCHS = 3000           # Increased for better convergence
    SAVE_INTERVAL = 1000
    CHECKPOINT_PATH = "checkpoints/trajectory_diffusion_policy_3000.pth"
    SAVE_DIR = "checkpoints"
    
    # ==========================================
    # 1. Dataset & DataLoader
    # ==========================================
    print("[trajectory_distillation] Initializing Dataset...")
    dataset = TrajectoryDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 2. Models Initialization
    # ==========================================
    print("[trajectory_distillation] Loading Teacher Model...")
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
        teacher_inner.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"[trajectory_distillation] Loaded teacher from {CHECKPOINT_PATH}")
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

    # Target model is updated via EMA of Student
    target_ema = EMA(student_model, beta=0.999) # Higher beta for distillation stability

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
    scheduler.set_timesteps(TIMESTEPS) # Fix: define discretization grid
    
    loss_fn = ConsistencyDistillationLoss(
        teacher_model=teacher_inner,
        solver=scheduler
    )
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-4)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"[trajectory_distillation] Starting distillation for {EPOCHS} epochs...")
    
    with tqdm(range(1, EPOCHS + 1), desc="Distilling") as pbar:
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                obs = batch['obs'].to(DEVICE)
                actions = batch['action'].to(DEVICE)
                
                # Sample t and t_next
                t_idx = torch.randint(0, TIMESTEPS - 1, (1,), device=DEVICE)
                t_indices = t_idx.repeat(obs.shape[0])
                t_next_indices = t_indices + 1
                
                loss = loss_fn(
                    student_model=student_model,
                    target_model=target_model,
                    observations=obs,
                    actions=actions,
                    t=t_indices,
                    t_next=t_next_indices
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                target_ema.update(student_model)
                target_ema.copy_to(target_model)
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix({"Loss": f"{avg_loss:.6f}"})
            
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_path = os.path.join(SAVE_DIR, f"trajectory_consistency_policy_{epoch}.pth")
            torch.save(student_model.state_dict(), checkpoint_path)
            torch.save(student_model.state_dict(), os.path.join(SAVE_DIR, "trajectory_consistency_policy.pth"))
            print(f"[trajectory_distillation] Saved student model to {checkpoint_path}")

    print("[trajectory_distillation] Distillation completed.")

if __name__ == "__main__":
    train_consistency()

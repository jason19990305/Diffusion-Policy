import os, torch, copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDIMScheduler

from utils.noise_predictor import DiffusionPolicy, EMA
from utils.consistency import ConsistencyPolicy
from utils.consistency_loss import ConsistencyDistillationLoss
from point_maze_dataset import PointMazeDataset

def train_consistency():
    # 0. Setup & Hyperparameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TIMESTEPS, EMBED_DIM, NUM_HEADS, MLP_RATIO, DEPTH = 100, 256, 8, 4.0, 8
    PRED_HORIZON, OBS_HORIZON = 32, 2
    BATCH_SIZE, LR, TOTAL_STEPS = 1024, 2e-4, 10000
    LOG_INTERVAL, SAVE_INTERVAL = 100, 2000
    CHECKPOINT_PATH, SAVE_DIR = "checkpoints/point_maze_diffusion.pth", "checkpoints"
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = PointMazeDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Models Initialization
    # Initialize and load Teacher
    teacher_inner = DiffusionPolicy(
        action_dim=dataset.action_dim, state_dim=dataset.state_dim,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, num_blocks=DEPTH
    ).to(DEVICE)
    
    teacher_inner.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    teacher_inner.eval().requires_grad_(False)

    # Initialize Student (from Teacher weights)
    student_inner = copy.deepcopy(teacher_inner).requires_grad_(True)
    student_model = ConsistencyPolicy(student_inner, sigma_data=0.1).to(DEVICE)
    
    # Initialize Target as a copy of Student
    target_model = copy.deepcopy(student_model).requires_grad_(False)
    target_ema = EMA(student_model, beta=0.999)

    # 3. Solver, Loss & Optimizer
    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, set_alpha_to_one=True, prediction_type="epsilon"
    )
    scheduler.set_timesteps(TIMESTEPS) 
    
    loss_fn = ConsistencyDistillationLoss(teacher_model=teacher_inner, solver=scheduler)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    print(f"Starting distillation on {DEVICE}...")
    global_step = 0
    pbar = tqdm(total=TOTAL_STEPS, desc="Distilling")
    
    while global_step < TOTAL_STEPS:
        for batch in dataloader:
            if global_step >= TOTAL_STEPS: break
                
            obs, actions = batch['obs'].to(DEVICE), batch['action'].to(DEVICE)
            
            # Sample t for the batch
            t_indices = torch.randint(0, TIMESTEPS - 1, (obs.shape[0],), device=DEVICE)
            t_next_indices = t_indices + 1
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = loss_fn(
                    student_model=student_model, target_model=target_model,
                    observations=obs, actions=actions,
                    t=t_indices, t_next=t_next_indices
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update Target via EMA
            target_ema.update(student_model)
            target_ema.copy_to(target_model)
            
            global_step += 1
            pbar.update(1)
            
            if global_step % LOG_INTERVAL == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}", step=global_step)
                
            if global_step % SAVE_INTERVAL == 0 or global_step == TOTAL_STEPS:
                save_path = f"{SAVE_DIR}/point_maze_consistency_policy.pth"
                torch.save(student_model.state_dict(), save_path)
                # Also save a numbered checkpoint
                torch.save(student_model.state_dict(), f"{SAVE_DIR}/point_maze_consistency_step_{global_step}.pth")

    pbar.close()
    print("Distillation completed.")

if __name__ == "__main__":
    train_consistency()
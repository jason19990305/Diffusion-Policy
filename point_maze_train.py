import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from noise_predictor import DiffusionPolicy, EMA
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

if __name__ == "__main__":
    # --- 1. Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    TIMESTEPS = 100         # Total diffusion steps
    EMBED_DIM = 256         # Embedding dimension for the model
    NUM_HEADS = 8           # Number of attention heads
    MLP_RATIO = 4.0         # MLP expansion ratio
    PRED_HORIZON = 32       # Prediction horizon
    OBS_HORIZON = 2         # Observation horizon 
    
    # RTX 5070 TI Max Optimization
    BATCH_SIZE = 1024       # Maximized for high-end GPU
    TOTAL_STEPS = 30000    # Step-based training instead of epochs
    LOG_INTERVAL = 100      # Steps between logging
    SAVE_INTERVAL = 5000   # Steps between saving checkpoints
    
    DEPTH = 8               # Number of transformer blocks
    LR = 1e-4               # Learning rate
    
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 2. Dataset and DataLoader ---
    dataset = PointMazeDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    # Using multiple workers and pin_memory for maximum throughput
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    # --- 3. Model and EMA ---
    model = DiffusionPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_blocks=DEPTH
    ).to(DEVICE)
    
    model.train()
    ema_model = EMA(model, beta=0.995)
    
    # --- 4. Scheduler, Optimizer, and AMP ---
    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    
    # Latest torch.amp API (non-deprecated)
    scaler = torch.amp.GradScaler('cuda')
    
    # --- 5. Training Loop (Step-based) ---
    print(f"Starting Training for {TOTAL_STEPS} steps...")
    global_step = 0
    pbar = tqdm(total=TOTAL_STEPS, desc="Training Progress")
    
    while global_step < TOTAL_STEPS:
        for batch in dataloader:
            if global_step >= TOTAL_STEPS:
                break
                
            obs = batch["obs"].to(DEVICE, non_blocking=True)
            actions = batch["action"].to(DEVICE, non_blocking=True)
            
            # 5.1 Sample diffusion steps
            k = torch.randint(0, TIMESTEPS, (obs.shape[0],), device=DEVICE)
            
            # 5.2 Add noise
            noise = torch.randn_like(actions)
            noised_actions = scheduler.add_noise(actions, noise, k)
            
            # 5.3 Predict noise with AMP
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                predicted_noise = model(observations=obs, diffusion_steps=k, noised_actions=noised_actions)
                loss = mse_loss(predicted_noise, noise)
            
            # 5.4 Backprop with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 5.5 Update EMA
            ema_model.update(model)
            
            global_step += 1
            pbar.update(1)
            
            if global_step % LOG_INTERVAL == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.6f}", "Step": global_step})
                
            if global_step % SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(SAVE_DIR, f"point_maze_diffusion_step_{global_step}.pth")
                ema_model.save_pretrained(checkpoint_path)
                # Keep a 'latest' symlink or copy
                ema_model.save_pretrained(os.path.join(SAVE_DIR, "point_maze_diffusion.pth"))

    pbar.close()
    print(f"Training completed. Final model saved to {os.path.join(SAVE_DIR, 'point_maze_diffusion.pth')}")

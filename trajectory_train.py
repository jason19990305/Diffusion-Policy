from trajectory_plot import generate_trajectory
from noise_predictor import DiffusionPolicy
from noise_predictor import EMA
from diffusers import DDIMScheduler
from trajectory_dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os



if __name__ == "__main__":
    # ==========================================
    # 0. Setup & Hyperparameters
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trajectory_train] Using device: {DEVICE}")

    # Hyperparameters for Diffusion Policy
    TIMESTEPS = 100         # Total diffusion steps
    EMBED_DIM = 256         # Embedding dimension for the model
    NUM_HEADS = 8           # Number of attention heads
    MLP_RATIO = 4.0         # MLP expansion ratio
    PRED_HORIZON = 16       # Prediction horizon for training samples
    OBS_HORIZON = 8         # Observation horizon 
    BATCH_SIZE = 32         # Batch size for training
    EPOCHS = 3000           # Number of epochs to train 
    DEPTH = 4               # Number of transformer blocks
    LR = 1e-4               # Learning rate for optimizer
    
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ==========================================
    # 1. Dataset & DataLoader
    # ==========================================
    print("[trajectory_train] Initializing Dataset and DataLoader...")
    trajectory_dataset = TrajectoryDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    dataloader = DataLoader(trajectory_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[trajectory_train] Dataset size: {len(trajectory_dataset)} | Batches/epoch: {len(dataloader)}")
    
    # ==========================================
    # 2. Model & EMA Definition
    # ==========================================
    print("[trajectory_train] Initializing Model and EMA...")
    model = DiffusionPolicy(
        action_dim=trajectory_dataset.action_dim,
        state_dim=trajectory_dataset.state_dim,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_blocks=DEPTH
    ).to(DEVICE)
    model.train()  
    ema_model = EMA(model, beta=0.995)
    
    # ==========================================
    # 3. Optimizer, Scheduler & Diffusion Setup
    # ==========================================
    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,   
        steps_offset=0,                         
        prediction_type="epsilon"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse_loss = torch.nn.MSELoss()
    
    # ==========================================
    # 4. Training Loop
    # ==========================================
    print(f"[trajectory_train] Starting training for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} Training")

        for step, batch in enumerate(pbar):
            actions = batch['action'].to(DEVICE)  
            obs = batch['obs'].to(DEVICE)  
            
            k = torch.randint(0, TIMESTEPS, (actions.shape[0],), device=actions.device)  
            noise = torch.randn_like(actions)  
            noised_actions = scheduler.add_noise(actions, noise, k) 
            
            predicted_noise = model(observations=obs, diffusion_steps=k, noised_actions=noised_actions)
            loss = mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_model.update(model)
            
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
        
        if epoch % 1000 == 0 or epoch == EPOCHS:
            checkpoint_path = os.path.join(SAVE_DIR, f"trajectory_diffusion_policy_{epoch}.pth")
            ema_model.save_pretrained(checkpoint_path)
            tqdm.write(f"[trajectory_train] Checkpoint saved at epoch {epoch} -> {checkpoint_path}")

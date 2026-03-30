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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters for Diffusion Policy
    TIMESTEPS = 100         # Total diffusion steps
    EMBED_DIM = 256         # Embedding dimension for the model
    NUM_HEADS = 8           # Number of attention heads
    MLP_RATIO = 4.0         # MLP expansion ratio
    PRED_HORIZON = 16       # Prediction horizon for training samples
    OBS_HORIZON = 8         # Observation horizon 
    BATCH_SIZE = 200         # Batch size for training
    EPOCHS = 3000           # Number of epochs to train 
    DEPTH = 4               # Number of transformer blocks
    LR = 1e-4               # Learning rate for optimizer
    
    # create checkpoint
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Get the trajectory dataset
    trajectory_dataset = TrajectoryDataset(pred_horizon=PRED_HORIZON,obs_horizon=OBS_HORIZON)
    # Create a DataLoader for batching
    dataloader = DataLoader(trajectory_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("")
    
    # Initialize the Diffusion Policy model
    model = DiffusionPolicy(action_dim=trajectory_dataset.action_dim,
                            state_dim=trajectory_dataset.state_dim,
                            embed_dim=EMBED_DIM,
                            num_heads=NUM_HEADS,
                            mlp_ratio=MLP_RATIO,
                            num_blocks=DEPTH).to(DEVICE)
    model.train()  # Set model to training mode
    # EMA to track the exponential moving average of the model parameters
    ema_model = EMA(model, beta=0.995)
    
    # Initialize the DDIM Scheduler for diffusion process
    scheduler = DDIMScheduler(num_train_timesteps=TIMESTEPS,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              set_alpha_to_one=True,   
                              steps_offset=0,                         
                              prediction_type="epsilon")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # loss function
    mse_loss = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
    
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS} Training")

        for step, batch in enumerate(pbar):
            # 1. Get the batch of action sequences
            actions = batch['action'].to(DEVICE)  
            obs = batch['obs'].to(DEVICE)  
            
            # 2. Sample random diffusion steps for each sequence in the batch
            k = torch.randint(0, TIMESTEPS, (actions.shape[0],), device=actions.device)  # shape: (batch_size,)
            
            # 3. Add noise to the actions according to the sampled diffusion steps
            noise = torch.randn_like(actions)  
            noised_actions = scheduler.add_noise(actions, noise, k) 
            
            # 4. Predict the noise using the Diffusion Policy model
            predicted_noise = model(observations=obs, diffusion_steps=k, noised_actions=noised_actions)
            
            # 5. Compute the loss between the predicted noise and the true noise
            loss = mse_loss(predicted_noise, noise)
            # 6. Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_model.update(model)
            pbar.set_postfix({"Epoch": epoch, "Loss": loss.item()})
        
        if epoch % 1000 == 0 or epoch == EPOCHS:
            checkpoint_path = os.path.join(SAVE_DIR, f"trajectory_diffusion_policy_{epoch}.pth")
            ema_model.save_pretrained(checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
        
    
    
    
    

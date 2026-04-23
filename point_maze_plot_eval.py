import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import collections
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

from utils.ensembling import NumpyTemporalEnsembler

def plot_eval(num_episodes=5):
    # --- 1. Hyperparameters & Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = "checkpoints/point_maze_diffusion.pth"
    
    # Model Architecture & Diffusion Params
    TIMESTEPS = 100         # Total diffusion steps
    INFERENCE_STEPS = 50    # Sampling steps for plotting accuracy
    EMBED_DIM = 256         # Embedding dimension for the model
    NUM_HEADS = 8           # Number of attention heads
    MLP_RATIO = 4.0         # MLP expansion ratio
    
    # Horizon Settings
    PRED_HORIZON = 64       # Total prediction horizon (model output length)
    OBS_HORIZON = 2         # Observation horizon (input length)
    ACTION_HORIZON = 4      # Execution steps per inference cycle
    
    ACTION_DIM = 2
    STATE_DIM = 6
    MAX_ENV_STEPS = 500
    SEED = 42

    # Calculate the effective future horizon for the Ensembler
    FUTURE_HORIZON = PRED_HORIZON - OBS_HORIZON + 1

    # --- 2. Load Components ---
    dataset = PointMazeDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    
    model = DiffusionPolicy(
        action_dim=ACTION_DIM, 
        state_dim=STATE_DIM, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        num_blocks=8
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS, 
        beta_schedule="squaredcos_cap_v2", 
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(INFERENCE_STEPS)

    env = gym.make('PointMaze_Large-v3', render_mode=None)
    all_trajectories = []
    
    # --- 3. Inference Loop ---
    for ep in range(num_episodes):
        print(f"Sampling Episode {ep+1}/{num_episodes}...")
        obs, _ = env.reset(seed=SEED)
        
        # Initialize observation buffer and ensembler
        curr_state = np.concatenate([obs['observation'], obs['desired_goal']])
        obs_buffer = collections.deque([curr_state] * OBS_HORIZON, maxlen=OBS_HORIZON)
        ensembler = NumpyTemporalEnsembler(FUTURE_HORIZON, ACTION_DIM)
        
        trajectory = []
        done = False
        
        for _ in range(MAX_ENV_STEPS):
            # A. Prepare Observation Tensor
            obs_seq = np.stack(obs_buffer)
            obs_normalized = dataset.state_normalizer.normalize(obs_seq)
            obs_tensor = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(DEVICE)
            
            # B. Diffusion Sampling
            noised_actions = torch.randn((1, PRED_HORIZON, ACTION_DIM), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise_pred = model(
                        obs_tensor, 
                        torch.full((1,), t, device=DEVICE, dtype=torch.long), 
                        noised_actions
                    )
                    noised_actions = scheduler.step(noise_pred, t, noised_actions).prev_sample
            
            # C. Unnormalize and Temporal Ensembling
            action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())
            future_actions = action_seq[OBS_HORIZON-1:] 
            
            ensembler.update(future_actions)
            exec_actions = ensembler.get_and_shift_actions(ACTION_HORIZON)
            
            # D. Execute Steps
            for action in exec_actions:
                obs, _, term, trunc, _ = env.step(action)
                
                # Update buffer and trajectory
                curr_state = np.concatenate([obs['observation'], obs['desired_goal']])
                obs_buffer.append(curr_state)
                trajectory.append(obs['achieved_goal'].copy())
                
                # Check termination (distance < 0.5)
                dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                if dist < 0.5 or term or trunc:
                    done = True
                    break
            if done: break
            
        all_trajectories.append(np.array(trajectory))

    # --- 4. Plotting Result ---
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
    
    # Plot goal location
    goal = obs['desired_goal']
    plt.scatter(goal[0], goal[1], marker='*', color='red', s=250, label='Goal', zorder=5)
    
    for i, traj in enumerate(all_trajectories):
        if len(traj) > 0:
            plt.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, label=f'Path {i+1}')
            plt.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=30)

    plt.title(f"Diffusion Policy Trajectories (PRED_H={PRED_HORIZON}, EXEC_H={ACTION_HORIZON})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig("diffusion_plot_eval.png")
    print("Plot saved to diffusion_plot_eval.png")
    plt.show()

if __name__ == "__main__":
    plot_eval()
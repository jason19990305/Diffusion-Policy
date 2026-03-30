import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import collections
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset
import os

def plot_eval(num_episodes=10):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "checkpoints/point_maze_diffusion.pth"

    # 1. Load Model and Stats
    dataset = PointMazeDataset(pred_horizon=16, obs_horizon=2)
    model = DiffusionPolicy(action_dim=2, state_dim=6, embed_dim=256, num_heads=8, num_blocks=8).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    scheduler.set_timesteps(50)

    # 2. Environment (No Rendering)
    env = gym.make('PointMaze_Large-v3', render_mode=None)
    all_trajectories = []
    
    SEED = 42
    for ep in range(num_episodes):
        print(f"Sampling {ep+1}/{num_episodes} with fixed Seed={SEED}...")
        obs, _ = env.reset(seed=SEED)
        current_state = np.concatenate([obs['observation'], obs['desired_goal']])
        obs_buffer = collections.deque([current_state] * 2, maxlen=2)
        trajectory = []
        
        for _ in range(500):
            obs_tensor = torch.from_numpy(dataset.state_normalizer.normalize(np.stack(obs_buffer))).float().unsqueeze(0).to(DEVICE)
            
            # Diffusion Inference
            noised_actions = torch.randn((1, 16, 2), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    pred_noise = model(obs_tensor, torch.full((1,), t, device=DEVICE, dtype=torch.long), noised_actions)
                    noised_actions = scheduler.step(pred_noise, t, noised_actions).prev_sample
            
            # Receding Horizon Execution (8 steps)
            action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())
            for i in range(8):
                obs, _, term, trunc, _ = env.step(action_seq[i])
                current_state = np.concatenate([obs['observation'], obs['desired_goal']])
                obs_buffer.append(current_state)
                trajectory.append(obs['achieved_goal'].copy())
                
                # Check if goal reached (distance < 0.5)
                dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                if dist < 0.5 or term or trunc:
                    term = True # Set term to True to break outer loop
                    break
            if term or trunc: break
        all_trajectories.append(np.array(trajectory))

    # 3. Trajectory Plotting (No Walls)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, num_episodes))
    
    # Plot goal (approximate from last obs)
    goal = obs['desired_goal']
    plt.scatter(goal[0], goal[1], marker='*', color='gold', s=300, label='Goal', zorder=5)
    
    for i, traj in enumerate(all_trajectories):
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.6, label=f'Path {i+1}')
        plt.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=20)

    plt.title("Diffusion Policy Multi-Trajectory Plot (PointMaze_Large-v3)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Ensure consistent scale to avoid visual distortion
    plt.tight_layout()
    
    plt.savefig("point_maze_inference.png")
    print("Plot saved to point_maze_inference.png")
    plt.show()

if __name__ == "__main__":
    plot_eval()

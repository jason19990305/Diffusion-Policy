import torch, numpy as np, gymnasium as gym, gymnasium_robotics, collections
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

def render_eval():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PointMazeDataset(pred_horizon=16, obs_horizon=2)
    model = DiffusionPolicy(action_dim=2, state_dim=6, embed_dim=256, num_heads=8, num_blocks=8).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/point_maze_diffusion.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    scheduler.set_timesteps(10)

    env = gym.make('PointMaze_Large-v3', render_mode='human')
    def get_state(o): return np.concatenate([o['observation'], o['desired_goal']])
    
    while True: # Episode loop
        obs, _ = env.reset()
        obs_buffer = collections.deque([get_state(obs)] * 2, maxlen=2)
        while True: # Control loop
            # Normalize and prepare observation
            obs_tensor = torch.from_numpy(dataset.state_normalizer.normalize(np.stack(obs_buffer))).float().unsqueeze(0).to(DEVICE)
            
            # Diffusion Inference
            noised_actions = torch.randn((1, 16, 2), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise = model(obs_tensor, torch.full((1,), t, device=DEVICE, dtype=torch.long), noised_actions)
                    noised_actions = scheduler.step(noise, t, noised_actions).prev_sample
            
            # Execute Receding Horizon (8 steps)
            action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())
            done = False
            for action in action_seq[:8]:
                obs, _, term, trunc, _ = env.step(action)
                obs_buffer.append(get_state(obs))
                if np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.3 or term or trunc:
                    print("Goal reached or episode ended. Resetting...")
                    done = True; break
            if done: break
    env.close()

if __name__ == "__main__":
    render_eval()

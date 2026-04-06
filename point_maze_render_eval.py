import torch, numpy as np, gymnasium as gym, gymnasium_robotics, collections
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

class TemporalEnsembling:
    def __init__(self, pred_horizon, action_dim):
        self.pred_horizon = pred_horizon
        self.action_sum = np.zeros((pred_horizon, action_dim))
        self.action_count = np.zeros((pred_horizon, 1))

    def update(self, predicted_action_seq):
        self.action_sum += predicted_action_seq
        self.action_count += 1

    def get_and_shift(self, execute_steps):
        counts = np.clip(self.action_count[:execute_steps], a_min=1, a_max=None)
        avg_actions = self.action_sum[:execute_steps] / counts
        new_sum = np.zeros_like(self.action_sum)
        new_count = np.zeros_like(self.action_count)
        if self.pred_horizon > execute_steps:
            new_sum[:-execute_steps] = self.action_sum[execute_steps:]
            new_count[:-execute_steps] = self.action_count[execute_steps:]
        self.action_sum = new_sum
        self.action_count = new_count
        return avg_actions

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
        
        # Initialize Temporal Ensembling
        obs_horizon, pred_horizon = 2, 16
        execute_steps = 4 # Reduced down to 4 for finer control
        effective_horizon = pred_horizon - obs_horizon + 1
        ensembler = TemporalEnsembling(effective_horizon, 2)
        
        while True: # Control loop
            # Normalize and prepare observation
            obs_tensor = torch.from_numpy(dataset.state_normalizer.normalize(np.stack(obs_buffer))).float().unsqueeze(0).to(DEVICE)
            
            # Diffusion Inference
            noised_actions = torch.randn((1, 16, 2), device=DEVICE)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise = model(obs_tensor, torch.full((1,), t, device=DEVICE, dtype=torch.long), noised_actions)
                    noised_actions = scheduler.step(noise, t, noised_actions).prev_sample
            
            # Temporal Ensembling Unnormalize and Slice
            action_seq = dataset.action_normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())
            
            # Slice future actions only
            start_idx = obs_horizon - 1
            future_actions = action_seq[start_idx:]
            
            # Ensemble and get smoothed actions
            ensembler.update(future_actions)
            smoothed_actions = ensembler.get_and_shift(execute_steps)
            
            done = False
            for action in smoothed_actions:
                obs, _, term, trunc, _ = env.step(action)
                obs_buffer.append(get_state(obs))
                if np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.3 or term or trunc:
                    print("Goal reached or episode ended. Resetting...")
                    done = True; break
            if done: break
    env.close()

if __name__ == "__main__":
    render_eval()

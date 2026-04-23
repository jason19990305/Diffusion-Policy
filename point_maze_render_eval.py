import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import collections
from utils.noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from point_maze_dataset import PointMazeDataset

from utils.ensembling import NumpyTemporalEnsembler

def render_eval():
    # --- 1. Hyperparameters & Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = "checkpoints/point_maze_diffusion.pth"
    
    # Model Architecture & Diffusion Params
    TIMESTEPS = 100         # Total diffusion steps
    INFERENCE_STEPS = 10    # Sampling steps for inference (can be lower than TIMESTEPS)
    EMBED_DIM = 256         # Embedding dimension for the model
    NUM_HEADS = 8           # Number of attention heads
    MLP_RATIO = 4.0         # MLP expansion ratio
    
    # Horizon Settings
    PRED_HORIZON = 32       # Total prediction horizon (model output length)
    OBS_HORIZON = 2         # Observation horizon (input length)
    ACTION_HORIZON = 4      # Execution steps per inference cycle
    
    ACTION_DIM = 2
    STATE_DIM = 6

    # Calculate the effective future horizon for the Ensembler
    # (From current timestep to the end of the prediction)
    FUTURE_HORIZON = PRED_HORIZON - OBS_HORIZON + 1

    # --- 2. Load Components ---
    dataset = PointMazeDataset(pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON)
    
    model = DiffusionPolicy(
        action_dim=ACTION_DIM, 
        state_dim=STATE_DIM, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        num_blocks=8 # Based on depth/blocks in your architecture
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS, 
        beta_schedule="squaredcos_cap_v2", 
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(INFERENCE_STEPS)

    # --- 3. Environment & Inference Loop ---
    env = gym.make('PointMaze_Large-v3', render_mode='human')
    
    def get_state(o): 
        return np.concatenate([o['observation'], o['desired_goal']])
    
    print(f"Starting Evaluation (Action Horizon: {ACTION_HORIZON})...")

    while True: # Episode Loop
        obs, _ = env.reset()
        obs_buffer = collections.deque([get_state(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON)
        ensembler = NumpyTemporalEnsembler(FUTURE_HORIZON, ACTION_DIM)
        
        while True: # Control Loop
            # A. Prepare Observation Tensor
            obs_seq = np.stack(obs_buffer)
            obs_normalized = dataset.state_normalizer.normalize(obs_seq)
            obs_tensor = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(DEVICE)
            
            # B. Diffusion Inference
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
            
            # Slice actions starting from the current timestep (index = OBS_HORIZON - 1)
            future_actions = action_seq[OBS_HORIZON-1:] 
            
            ensembler.update(future_actions)
            exec_actions = ensembler.get_and_shift_actions(ACTION_HORIZON)
            
            # D. Execute Action Sequence
            done = False
            for action in exec_actions:
                obs, _, term, trunc, _ = env.step(action)
                
                # Update observation buffer
                obs_buffer.append(get_state(obs))
                
                # Check termination conditions
                dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                if dist < 0.3 or term or trunc:
                    print(f"Episode Finished. Goal Distance: {dist:.3f}")
                    done = True
                    break
            
            if done: break

if __name__ == "__main__":
    render_eval()
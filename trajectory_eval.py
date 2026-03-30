import torch
import numpy as np
import collections
import matplotlib.pyplot as plt
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from trajectory_dataset import Normalization
from trajectory_plot import generate_trajectory_forward

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Parameters (strictly matching trajectory_gif_eval.py) ---
    TIMESTEPS = 100
    ACTION_DIM = 2
    STATE_DIM = 2
    EMBED_DIM = 256
    NUM_HEADS = 8
    MLP_RATIO = 4.0
    OBS_HORIZON = 8
    PRED_HORIZON = 16
    ACTION_HORIZON = 8
    INFERENCE_STEPS = 50
    MAX_EPISODE_STEPS = 100
    CHECKPOINT_PATH = "checkpoints/trajectory_diffusion_policy_3000.pth"

    # --- 2. Initialize Model and Scheduler ---
    model = DiffusionPolicy(action_dim=ACTION_DIM, state_dim=STATE_DIM,
                            embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                            mlp_ratio=MLP_RATIO, num_blocks=4).to(DEVICE)
    
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=TIMESTEPS,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, set_alpha_to_one=True,
                              prediction_type="epsilon")
    scheduler.set_timesteps(INFERENCE_STEPS)
    print(scheduler.timesteps)

    # --- 3. Setup Inference environment ---
    reference_traj = generate_trajectory_forward(num_steps=100)
    normalizer = Normalization(reference_traj)
    
    current_state = reference_traj[0].copy()
    # Observation buffer (window of past states)
    obs_buffer = collections.deque([current_state.copy()] * OBS_HORIZON, maxlen=OBS_HORIZON)
    actual_trajectory = [current_state.copy()]

    # --- 4. Closed-loop Inference ---
    print(f"Starting closed-loop inference for {MAX_EPISODE_STEPS} steps...")
    step_count = 0
    while step_count < MAX_EPISODE_STEPS:
        # Prepare Observation (shape: 1, OBS_HORIZON, STATE_DIM)
        obs_np = np.stack(obs_buffer)
        obs_norm = normalizer.normalize(obs_np)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(DEVICE)
        
        # Action Prediction (Gaussian Noise -> Predicted Trajectory)
        noised_actions = torch.randn((1, PRED_HORIZON, ACTION_DIM), device=DEVICE)
        with torch.no_grad():
            for k in scheduler.timesteps:
                t_batch = torch.full((1,), k, device=DEVICE, dtype=torch.long)
                predicted_noise = model(observations=obs_tensor, 
                                        diffusion_steps=t_batch, 
                                        noised_actions=noised_actions)
                noised_actions = scheduler.step(model_output=predicted_noise, 
                                                timestep=k, 
                                                sample=noised_actions).prev_sample

        # Post-process: Unnormalize
        predicted_action_seq = normalizer.unnormalize(noised_actions.squeeze(0).cpu().numpy())

        # Execute Actions (Receding Horizon)
        for i in range(ACTION_HORIZON):
            if step_count >= MAX_EPISODE_STEPS:
                break
            
            # Use predicted point as next state (simple simulator)
            current_state = predicted_action_seq[i]
            actual_trajectory.append(current_state.copy())
            obs_buffer.append(current_state.copy())
            step_count += 1
            
        print(f"Progress: {step_count}/{MAX_EPISODE_STEPS}", end='\r')

    # --- 5. Plotting and Comparison ---
    actual_trajectory = np.array(actual_trajectory)
    plt.figure(figsize=(8, 6))
    # Target Trajectory
    plt.plot(reference_traj[:, 0], reference_traj[:, 1], color='gray', linestyle='--', label='Target Trajectory', alpha=0.6)
    # Predicted/Actual Path
    plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], color='blue', label='Generated Path', linewidth=2)
    # Start/End Markers
    plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], color='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], color='red', marker='x', s=100, label='End', zorder=5)
    
    plt.title("Trajectory Evaluation: Target vs. Generated (Diffusion Policy)")
    plt.xlabel("X coordinates"), plt.ylabel("Y coordinates")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    
    output_path = "trajectory_inference.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"\nInference completed. Comparison plot saved as {output_path}")

if __name__ == "__main__":
    main()

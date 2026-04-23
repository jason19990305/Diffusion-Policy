import torch
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assume these are imported from your actual project
from noise_predictor import DiffusionPolicy
from diffusers import DDIMScheduler
from utils.normalization import NumpyNormalizer
from trajectory_plot import generate_trajectory_forward
from utils.ensembling import NumpyTemporalEnsembler

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Hyperparameters ---
    TIMESTEPS = 100         
    ACTION_DIM = 2          
    STATE_DIM = 2           # Changed from 0 to 2 to accept (x, y) observations
    EMBED_DIM = 256         
    NUM_HEADS = 8           
    MLP_RATIO = 4.0         
    
    # Receding Horizon parameters
    OBS_HORIZON = 8         # Increased to 8 to match training settings
    PRED_HORIZON = 16       # Usually 16 is standard for 2D trajectories
    ACTION_HORIZON = 2      # Execute 2 steps, then replan (reduced for better ensembling)
    
    INFERENCE_STEPS = 50
    MAX_EPISODE_STEPS = 100 # We want the agent to walk for 100 steps
    CHECKPOINT_PATH = "checkpoints/trajectory_diffusion_policy_3000.pth"

    # --- 2. Initialize Model and Scheduler ---
    model = DiffusionPolicy(action_dim=ACTION_DIM, state_dim=STATE_DIM,
                            embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                            mlp_ratio=MLP_RATIO, num_blocks=4).to(DEVICE)
    
    # Load weights (Ensure the model parses the trained weights)
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=TIMESTEPS,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, set_alpha_to_one=True,   
                              steps_offset=0, prediction_type="epsilon")
    scheduler.set_timesteps(INFERENCE_STEPS)

    # --- 3. Setup the "Environment" ---
    reference_traj = generate_trajectory_forward(num_steps=100)
    # Create a Normalizer to perform normalization/denormalization
    normalizer = NumpyNormalizer(reference_traj)
    
    # Our target starts at (0, 0) based on sin(0), sin(2*0)
    current_state = reference_traj[0].copy() 
    
    # Use a deque to maintain the observation window easily
    obs_buffer = collections.deque(maxlen=OBS_HORIZON)
    # Pad the beginning with the initial state
    for _ in range(OBS_HORIZON):
        noisy_state = current_state + np.random.normal(0, 1e-4, size=STATE_DIM)
        obs_buffer.append(noisy_state)

    # To record the actual path taken by the agent
    actual_trajectory = [current_state.copy()]
    
    # Initialize Temporal Ensembling
    temporal_ensembler = NumpyTemporalEnsembler(pred_horizon=PRED_HORIZON, action_dim=ACTION_DIM)
    
    print("Starting closed-loop inference...")
    step_count = 0
    actual_path_len = 1
    animation_frames = []
    
    # --- 4. Closed-loop Control Loop ---
    while step_count < MAX_EPISODE_STEPS:
        
        # 4.1 Prepare Observation (shape: 1, OBS_HORIZON, STATE_DIM)
        obs_np = np.stack(obs_buffer) # shape: (OBS_HORIZON, STATE_DIM)
        # Normalize the observation
        obs_np_norm = normalizer.normalize(obs_np)
        # convert to tensor
        obs_tensor = torch.from_numpy(obs_np_norm).float().unsqueeze(0).to(DEVICE)
        
        # 4.2 Initialize pure noise for actions
        trajectory_shape = (1, PRED_HORIZON, ACTION_DIM)
        noised_actions = torch.randn(trajectory_shape, device=DEVICE)

        # 4.3 Reverse Diffusion Process (Action Prediction)
        with torch.no_grad():
            for step_idx, t in enumerate(scheduler.timesteps):
                t_batch = torch.full((1,), t, device=DEVICE, dtype=torch.long)
                # Now we feed the actual observation into the model!
                predicted_noise = model(observations=obs_tensor, 
                                        diffusion_steps=t_batch, 
                                        noised_actions=noised_actions)
                
                step_result = scheduler.step(model_output=predicted_noise, 
                                             timestep=t, 
                                             sample=noised_actions)
                noised_actions = step_result.prev_sample

                # Record diffusion process for animation (e.g. every 2 steps to make the green scatter move slower/smoother)
                if step_idx % 2 == 0 or step_idx == len(scheduler.timesteps) - 1:
                    interm_seq_norm = noised_actions.squeeze(0).cpu().numpy()
                    interm_seq = normalizer.unnormalize(interm_seq_norm)
                    animation_frames.append({
                        'agent_path_len': actual_path_len,
                        'diffusion_points': interm_seq
                    })

        # Remove batch dimension and convert to numpy: shape (PRED_HORIZON, ACTION_DIM)
        predicted_action_seq_norm = noised_actions.squeeze(0).cpu().numpy()
        
        # Unnormalize to get real coordinates
        predicted_action_seq = normalizer.unnormalize(predicted_action_seq_norm)

        # Add the predicted trajectory to Temporal Ensembling for averaging
        temporal_ensembler.update(predicted_action_seq)
        
        # Retrieve the smoothed ACTION_HORIZON actions
        smoothed_actions = temporal_ensembler.get_and_shift_actions(ACTION_HORIZON)

        # 4.4 Execute actions (Receding Horizon)
        # We only execute the first ACTION_HORIZON steps from the prediction
        for i in range(ACTION_HORIZON):
            if step_count >= MAX_EPISODE_STEPS:
                break
                
            # Use the averaged action from Temporal Ensembling
            action = smoothed_actions[i]
            current_state = action 
            
            # Update history and buffers
            actual_trajectory.append(current_state.copy())
            obs_buffer.append(current_state.copy()) # Push new state, oldest pops out automatically
            
            actual_path_len += 1
            step_count += 1
            
            # Record execution frame
            animation_frames.append({
                'agent_path_len': actual_path_len,
                'diffusion_points': predicted_action_seq
            })
            
        print(f"Executed {step_count}/{MAX_EPISODE_STEPS} steps...")

    print("Inference completed! Generating GIF...")

    # --- 5. Animate and Save as GIF ---
    actual_trajectory = np.array(actual_trajectory)
    x_coords = actual_trajectory[:, 0]
    y_coords = actual_trajectory[:, 1]
    
    # Target trajectory for background
    target_x = reference_traj[:, 0]
    target_y = reference_traj[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Diffusion Policy Closed-Loop Control")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.set_aspect('equal')

    # Target trajectory for background
    ax.plot(target_x, target_y, color='lightgray', linestyle='--', linewidth=2, label='Target Trajectory')

    # Elements to animate
    line, = ax.plot([],[], marker='o', color='blue', linestyle='-', markersize=4, label='Agent Path')
    current_pos, = ax.plot([],[], marker='*', color='red', markersize=12, label='Current Pos')
    diffusion_scatter, = ax.plot([], [], marker='o', color='green', alpha=0.5, linestyle='None', markersize=3, label='Diffusion Prediction')
    ax.legend(loc="upper right")

    def init():
        line.set_data([],[])
        current_pos.set_data([],[])
        diffusion_scatter.set_data([],[])
        return line, current_pos, diffusion_scatter

    def update(frame_idx):
        frame = animation_frames[frame_idx]
        length = frame['agent_path_len']
        line.set_data(x_coords[:length], y_coords[:length])
        current_pos.set_data([x_coords[length-1]], [y_coords[length-1]])
        
        diff_pts = frame['diffusion_points']
        if diff_pts is not None:
            diffusion_scatter.set_data(diff_pts[:, 0], diff_pts[:, 1])
        else:
            diffusion_scatter.set_data([], [])
            
        return line, current_pos, diffusion_scatter

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(animation_frames),
                        init_func=init, blit=True, interval=33) # interval=33ms per frame (~30 fps)

    # Save as GIF using Pillow writer 
    gif_path = "diffusion_policy_trajectory.gif"
    ani.save(gif_path, writer='pillow', fps=30, 
             savefig_kwargs={'transparent': False, 'facecolor': 'white'})
    
    from PIL import Image, ImageSequence
    img = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    
    durations = [33] * len(frames)
    durations[-1] = 2000  # Last frame display for 2 seconds
    
    frames[0].save(fp=gif_path, format='GIF', append_images=frames[1:],
                   save_all=True, duration=durations, loop=0)
    
    print(f"GIF saved successfully at {gif_path}!")
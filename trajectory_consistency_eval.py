import torch
import numpy as np
import collections
import matplotlib.pyplot as plt
import os

from utils.noise_predictor import DiffusionPolicy
from utils.consistency import ConsistencyPolicy, ConsistencySampler
from utils.normalization import NumpyNormalizer
from trajectory_plot import generate_trajectory_forward, generate_trajectory_reverse
from utils.ensembling import NumpyTemporalEnsembler

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Parameters ---
    ACTION_DIM = 2
    STATE_DIM = 2
    EMBED_DIM = 256
    NUM_HEADS = 8
    MLP_RATIO = 4.0
    OBS_HORIZON = 8
    PRED_HORIZON = 16
    DEPTH = 4
    
    # Execution horizon (how many steps to take before replanning)
    ACTION_HORIZON = 2 
    MAX_EPISODE_STEPS = 100
    
    # We will test 2-step consistency policy
    CONSISTENCY_STEPS = 2
    CHECKPOINT_PATH = "checkpoints/trajectory_consistency_policy_1000.pth"

    # --- 2. Initialize Model ---
    inner_model = DiffusionPolicy(
        action_dim=ACTION_DIM, state_dim=STATE_DIM,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO, num_blocks=DEPTH
    ).to(DEVICE)
    
    model = ConsistencyPolicy(inner_model).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded consistency policy from {CHECKPOINT_PATH}")
    else:
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
        return
        
    model.eval()
    sampler = ConsistencySampler(model)

    # --- 3. Setup Inference environment ---
    reference_traj = generate_trajectory_forward(num_steps=100)
    # Combine forward and reverse to initialize normalizer
    combined_np = np.concatenate([reference_traj, generate_trajectory_reverse(num_steps=100)], axis=0)
    normalizer = NumpyNormalizer(combined_np)
    
    # Initialize observation buffer with the starting point
    current_state = reference_traj[0].copy()
    obs_buffer = collections.deque([current_state.copy()] * OBS_HORIZON, maxlen=OBS_HORIZON)
    actual_trajectory = [current_state.copy()]

    # Initialize Temporal Ensembling
    temporal_ensembler = NumpyTemporalEnsembler(pred_horizon=PRED_HORIZON, action_dim=ACTION_DIM)

    # --- 4. Closed-loop Inference ---
    print(f"Starting closed-loop inference using {CONSISTENCY_STEPS}-Step Consistency Policy...")
    step_count = 0
    
    while step_count < MAX_EPISODE_STEPS:
        # Prepare Observation
        obs_np = np.stack(obs_buffer)
        obs_norm = normalizer.normalize(obs_np)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(DEVICE)
        
        # Action Prediction (2-Step Consistency)
        # Since 2-step is highly stable, we don't need multi-sample KDE anymore.
        # Just a single sample is enough, making it lightning fast!
        best_action_seq = sampler.sample(
            observations=obs_tensor,
            pred_horizon=PRED_HORIZON,
            action_dim=ACTION_DIM,
            steps=CONSISTENCY_STEPS,
            sigma_start=1.0
        )

        # Post-process: Unnormalize
        predicted_action_seq = normalizer.unnormalize(best_action_seq.squeeze(0).cpu().numpy())

        # Add to Temporal Ensembling
        temporal_ensembler.update(predicted_action_seq)
        
        # Retrieve the smoothed actions
        smoothed_actions = temporal_ensembler.get_and_shift_actions(ACTION_HORIZON)

        # Execute Actions (Receding Horizon)
        for i in range(ACTION_HORIZON):
            if step_count >= MAX_EPISODE_STEPS:
                break
            
            # Action becomes the next state
            current_state = smoothed_actions[i]
            actual_trajectory.append(current_state.copy())
            obs_buffer.append(current_state.copy())
            step_count += 1
            
        print(f"Progress: {step_count}/{MAX_EPISODE_STEPS}", end='\r')

    # --- 5. Plotting and Comparison ---
    actual_trajectory = np.array(actual_trajectory)
    plt.figure(figsize=(10, 6))
    
    # Plot background paths (both forward and reverse)
    plt.plot(reference_traj[:, 0], reference_traj[:, 1], color='green', linestyle='--', alpha=0.3, label='Target 1')
    ref_rev = generate_trajectory_reverse()
    plt.plot(ref_rev[:, 0], ref_rev[:, 1], color='green', linestyle='--', alpha=0.3, label='Target 2')
    
    # Predicted/Actual Path
    plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], color='red', label=f'Closed-Loop Rollout ({CONSISTENCY_STEPS}-Step)', linewidth=2)
    
    # Start/End Markers
    plt.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], color='blue', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], color='black', marker='x', s=100, label='End', zorder=5)
    
    plt.title(f"Consistency Policy Rollout (Steps={CONSISTENCY_STEPS}, Ensembling)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    
    output_path = f"trajectory_consistency_rollout_{CONSISTENCY_STEPS}step.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"\nInference completed. Rollout plot saved as {output_path}")

if __name__ == "__main__":
    main()

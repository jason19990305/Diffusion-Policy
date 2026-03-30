import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory_forward(num_steps=100):
    # 1. generate time sequence t (from 0 to 2*pi)
    t = np.linspace(0, 2 * np.pi, num_steps)
    # Mode 1
    x = np.sin(t)
    y = np.sin(2 * t)
    return np.stack([x, y], axis=-1)

def generate_trajectory_reverse(num_steps=100):
    # 1. generate time sequence t (from 0 to 2*pi)
    t = np.linspace(0, 2 * np.pi, num_steps)
    # Mode 2
    x = -np.sin(t)
    y = -np.sin(2 * t)
    return np.stack([x, y], axis=-1)


def generate_trajectory(num_steps=100):
    traj1 = generate_trajectory_forward(num_steps)
    traj2 = generate_trajectory_reverse(num_steps)
    return np.concatenate([traj1, traj2], axis=0)


if __name__ == "__main__":
    num_steps = 100
    traj_fwd = generate_trajectory_forward(num_steps)
    traj_rev = generate_trajectory_reverse(num_steps)
    plt.figure(figsize=(8, 6))

    # Visualize Forward Trajectory
    #plt.plot(traj_fwd[:, 0], traj_fwd[:, 1], color='blue', linestyle='--', alpha=0.4, label='Path')
    #sc1 = plt.scatter(traj_fwd[:, 0], traj_fwd[:, 1], c=np.arange(num_steps), cmap='Blues', s=50, zorder=3)

    plt.plot(traj_rev[:, 0], traj_rev[:, 1], color='red', linestyle='--', alpha=0.4, label='Path')
    sc1 = plt.scatter(traj_rev[:, 0], traj_rev[:, 1], c=np.arange(num_steps), cmap='Reds', s=50, zorder=3)

  
    plt.title('Multi-modal 2D Action Trajectory for Diffusion Policy')
    plt.xlabel('Action Dim 0 (X)')
    plt.ylabel('Action Dim 1 (Y)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.show()
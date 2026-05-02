import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def generate_real_feature_map(size, obj_pos, sigma=1.0, noise_level=0.15):
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)
    dist_sq = (x - obj_pos[0])**2 + (y - obj_pos[1])**2
    blob = np.exp(-dist_sq / (2 * sigma**2))
    noise = np.random.rand(size, size) * noise_level
    return blob + noise

# Set parameters
size = 14
frames_count = 40  # Total frames; keep short to avoid large file size
obj_path_x = np.linspace(-0.6, 0.6, frames_count)
obj_path_y = np.sin(obj_path_x * 4) * 0.5  # S-shaped curve path

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=80)

def update(frame):
    ax1.clear()
    ax2.clear()
    
    current_pos = [obj_path_x[frame], obj_path_y[frame]]
    f_map = generate_real_feature_map(size, current_pos)
    
    # Comparison: Low Temperature vs High Temperature
    temps = [0.05, 1.0]
    titles = ["Precision Mode (Low T=0.05)", "Stability Mode (High T=1.0)"]
    
    for i, (t, ax, title) in enumerate(zip(temps, [ax1, ax2], titles)):
        # Compute Spatial Softmax
        exp_f = np.exp(f_map / t)
        weights = exp_f / np.sum(exp_f)
        
        grid_vals = np.linspace(-1, 1, size)
        pos_y, pos_x = np.meshgrid(grid_vals, grid_vals, indexing='ij')
        kx, ky = np.sum(pos_x * weights), np.sum(pos_y * weights)
        
        # Plotting
        ax.imshow(weights, extent=[-1, 1, 1, -1], cmap='magma', origin='upper')
        ax.plot(kx, ky, 'wo', markersize=10, markeredgecolor='cyan', label='Keypoint')
        ax.plot(current_pos[0], current_pos[1], 'rx', markersize=8, label='True Pos')
        
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(1.1, -1.1)
        if frame == 0: ax.legend(loc='upper right', fontsize='small')

# Create animation
print("Calculating and generating GIF, please wait...")
ani = FuncAnimation(fig, update, frames=frames_count, interval=100)

# Save as GIF
# fps=10 means 10 frames per second
writer = PillowWriter(fps=10)
ani.save("spatial_softmax_demo.gif", writer=writer)

plt.close() # Close plot to free memory
print("Finished! File saved as: spatial_softmax_demo.gif")
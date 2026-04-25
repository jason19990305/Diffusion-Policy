# Diffusion Policy (Transformer-based)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28%2B-green)](https://gymnasium.farama.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.20%2B-yellow)](https://huggingface.co/docs/diffusers/index)

HackMD Article : https://hackmd.io/@bGCXESmGSgeAArScMaBxLA/B1vYJqVu-g

This project implements a robotic behavior learning framework based on the **Diffusion Policy** architecture. It treats action sequence prediction as a conditional generative modeling problem, utilizing a **Transformer-based Diffusion Model** to learn complex multi-modal behaviors. The project supports various environments, ranging from simple 2D trajectories to high-dimensional visual manipulation tasks like ALOHA, and leverages **Temporal Ensembling** for smooth, robust action execution.

## 📂 Project Structure

```text
.
├── trajectory_train.py         # Training script for 2D trajectory generation
├── trajectory_eval.py          # Evaluation script for trajectory (Plotting results)
├── trajectory_gif_eval.py      # Generates GIF animations of trajectory denoising
├── point_maze_train.py         # Training script for Point Maze navigation
├── point_maze_plot_eval.py     # Evaluation script for Point Maze (State-based)
├── aloha_train.py              # Main training script for ALOHA manipulation (Vision-based)
├── aloha_render_eval.py        # Evaluation script for ALOHA with video rendering
├── utils/
│   ├── noise_predictor.py      # Core DiffusionPolicy (Transformer) & VisionEncoder
│   ├── ensembling.py           # Temporal Ensembling implementation (NumPy/PyTorch)
│   └── normalization.py        # Dataset normalization utilities
├── checkpoints/                # Directory for saving trained model weights
├── assets/                     # Directory for storing generated plots and images
└── README.md                   # Project documentation
```

## 🚀 Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** and a CUDA-capable GPU.

### 2. Install Dependencies
```bash
# Core dependencies
pip install torch torchvision numpy matplotlib tqdm gymnasium

# Diffusion and Vision dependencies
pip install diffusers transformers opencv-python
```

## 🖥️ Usage

### 1. Trajectory (2D Path Generation)
Train a model to generate smooth 2D trajectories.
```bash
# Training
python trajectory_train.py

# Evaluation & Plotting
python trajectory_eval.py

# Generate Visualization GIF
python trajectory_gif_eval.py
```

### 2. Point Maze (State-based Navigation)
Train a policy to navigate a point mass through a complex maze.
```bash
# Training
python point_maze_train.py

# Static Evaluation Plot
python point_maze_plot_eval.py

# Render Navigation Video
python point_maze_render_eval.py
```

### 3. ALOHA (Vision-based Manipulation)
Train a vision-based policy for robotic manipulation tasks using the ALOHA dataset.
```bash
# Training (with Mixed Precision & Gradient Checkpointing)
python aloha_train.py --batch_size 32 --total_steps 400000

# Evaluation & Video Generation
python aloha_render_eval.py --checkpoint checkpoints/aloha_diffusion.pth
```

## 💡 Technical Highlights

- **Diffusion Transformer (DiT) Backbone**:
  Replaces the traditional U-Net with a scalable **Transformer** architecture. Actions are treated as tokens, allowing the model to capture long-range temporal dependencies in action sequences more effectively.
  
- **Vision-based Control with Spatial Softmax**:
  The ALOHA policy integrates a **VisionEncoder** (Mini-ResNet) that utilizes **Spatial Softmax** to extract low-dimensional coordinate features (keypoints) from raw RGB images, providing a robust representation for manipulation tasks.

- **Temporal Ensembling**:
  Implements **Temporal Ensembling** to aggregate overlapping action predictions from multiple diffusion horizons. This significantly reduces jitter and ensures smooth, continuous robotic motion during closed-loop execution.

- **Advanced Diffusion Scheduling**:
  Utilizes the `DDIMScheduler` with a `squaredcos_cap_v2` beta schedule for both training and accelerated jump-step inference (e.g., 50-100 steps).

- **High-Performance Training**:
  Features **EMA (Exponential Moving Average)** for weight smoothing, **Mixed Precision (BFloat16)** for faster computation, and **Gradient Checkpointing** to enable training deep Transformers on consumer GPUs.

## Result
### Point Maze Navigation
![Point Maze Result](./assets/point_maze_navigation.png)

### Trajectory Denoising Process
![Trajectory GIF](./assets/trajectory_denoising.gif)

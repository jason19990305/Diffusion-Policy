# Diffusion Policy (Transformer-based)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28%2B-green)](https://gymnasium.farama.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.20%2B-yellow)](https://huggingface.co/docs/diffusers/index)

HackMD Article : https://hackmd.io/@bGCXESmGSgeAArScMaBxLA/rJHw7tG9Zl

This project implements a robotic behavior learning framework based on the **Diffusion Policy** architecture. It treats action sequence prediction as a conditional generative modeling problem, utilizing a **Transformer-based Diffusion Model** to learn complex multi-modal behaviors. The project supports various environments, ranging from simple 2D trajectories to high-dimensional visual manipulation tasks like ALOHA, and leverages **Temporal Ensembling** for smooth, robust action execution.

## 📂 Project Structure

```text
.
├── trajectory_train.py                   # Training script for 2D trajectory generation
├── trajectory_eval.py                    # Evaluation script for trajectory (Plotting results)
├── trajectory_gif_eval.py                # Generates GIF animations of trajectory denoising
├── trajectory_consistency_distillation.py # Distills 2D trajectory Diffusion Policy to Consistency Policy
├── trajectory_consistency_eval.py         # Evaluation script for trajectory using Consistency Policy
├── point_maze_train.py                   # Training script for Point Maze navigation
├── point_maze_plot_eval.py               # Evaluation script for Point Maze (State-based)
├── point_maze_render_eval.py             # Render navigation video for Point Maze (State-based)
├── point_maze_consistency_distillation.py # Distills Point Maze Diffusion Policy to Consistency Policy
├── point_maze_consistency_render_eval.py  # Render navigation video using Consistency Policy
├── aloha_train.py                        # Main training script for ALOHA manipulation (Vision-based)
├── aloha_render_eval.py                  # Evaluation script for ALOHA with video rendering
├── aloha_consistency_distillation.py     # Distills ALOHA Diffusion Policy to Consistency Policy
├── aloha_consistency_eval.py             # Evaluation script for ALOHA using Consistency Policy with rendering
├── utils/
│   ├── noise_predictor.py                # Core DiffusionPolicy (Transformer) & VisionEncoder
│   ├── consistency.py                    # Core ConsistencyPolicy wrapper and ConsistencySampler
│   ├── consistency_loss.py               # Consistency Distillation Loss implementation
│   ├── ensembling.py                     # Temporal Ensembling implementation (NumPy/PyTorch)
│   └── normalization.py                  # Dataset normalization utilities
├── checkpoints/                          # Directory for saving trained model weights
├── assets/                               # Directory for storing generated plots and images
└── README.md                             # Project documentation
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

### 4. Consistency Policy (Fast Inference via Distillation)
Train a Consistency Policy via distillation from a pre-trained Diffusion Policy teacher model for lightning-fast action generation (1-2 steps).

#### 2D Trajectory
```bash
# Distillation training (requires trained checkpoints/trajectory_diffusion_policy_3000.pth)
python trajectory_consistency_distillation.py

# Evaluation & Plotting
python trajectory_consistency_eval.py
```

#### Point Maze
```bash
# Distillation training (requires trained checkpoints/point_maze_diffusion.pth)
python point_maze_consistency_distillation.py

# Render Navigation Video (with GUI rendering)
python point_maze_consistency_render_eval.py
```

#### ALOHA
```bash
# Distillation training (requires trained checkpoints/aloha_diffusion_step_400000.pth)
python aloha_consistency_distillation.py --batch_size 32 --total_steps 100000

# Evaluation & Video Generation
python aloha_consistency_eval.py --checkpoint checkpoints/aloha_consistency_policy.pth
```

## 💡 Technical Highlights

- **Diffusion Transformer (DiT) Backbone**:
  Replaces the traditional U-Net with a scalable **Transformer** architecture. Actions are treated as tokens, allowing the model to capture long-range temporal dependencies in action sequences more effectively.
  
- **Vision-based Control with Spatial Softmax**:
  The ALOHA policy integrates a **VisionEncoder** (Mini-ResNet) that utilizes **Spatial Softmax** to extract low-dimensional coordinate features (keypoints) from raw RGB images, providing a robust representation for manipulation tasks.

- **Temporal Ensembling**:
  Implements **Temporal Ensembling** to aggregate overlapping action predictions from multiple diffusion horizons. This significantly reduces jitter and ensures smooth, continuous robotic motion during closed-loop execution.

- **Consistency Policy (Consistency Models) for Real-Time Control**:
  Implements a **ConsistencyPolicy** wrapper and **ConsistencyDistillationLoss** to distill pre-trained multi-step Diffusion Policies into Consistency Policies. This allows the agent to generate actions in only **1 to 2 inference steps** (instead of 50-100), drastically reducing control loop latency and enabling real-time execution on resource-constrained robotic platforms.

- **Advanced Diffusion Scheduling**:
  Utilizes the `DDIMScheduler` with a `squaredcos_cap_v2` beta schedule for both training and accelerated jump-step inference (e.g., 50-100 steps).

- **High-Performance Training**:
  Features **EMA (Exponential Moving Average)** for weight smoothing, **Mixed Precision (BFloat16)** for faster computation, and **Gradient Checkpointing** to enable training deep Transformers on consumer GPUs.

## Result



### Trajectory Denoising Process
<img width="800" height="600" alt="diffusion_policy_trajectory" src="https://github.com/user-attachments/assets/a4edbb55-e147-4227-93f3-eb44a1fed48b" />




### Point Maze Navigation
<img width="1462" height="1083" alt="image" src="https://github.com/user-attachments/assets/bbcd912b-80cb-4cbd-b243-1c65b0b60a42" />

<img width="528" height="357" alt="upload_6bb4d7ad565d722cfb8713de47033d4a" src="https://github.com/user-attachments/assets/3cb261bc-a3d1-409b-a361-66170b798dfc" />


### Aloha Transfer Cube

<img width="640" height="480" alt="eval_aloha_0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/1c08e48c-a753-45cf-b0ec-ab26a2973e86" />


#!/bin/bash

echo "🚀 Starting Final Corrected Setup for H100 (2026-04-04)..."

# 1. System Dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglib2.0-0 \
    libegl1-mesa libgles2-mesa \
    ffmpeg libsndfile1 git-lfs ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. Complete Cleanup
python -m pip uninstall -y xformers opencv-python opencv-contrib-python \
    opencv-python-headless numpy torch torchvision torchaudio lerobot

# 3. Install PyTorch
echo "🔥 Installing PyTorch 2.11.0..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install Base Tools & Build Dependencies (Fix for Python 3.12)
echo "🛠️ Installing Base Tools & Build Dependencies..."
python -m pip install --upgrade pip
# Must install setuptools and wheel first, otherwise old packages may fail to compile on Python 3.12
python -m pip install setuptools wheel "opencv-python-headless>=4.11.0.86" "numpy>=2.2.0"

# Install antlr4 (omegaconf dependency) separately, disabling build-isolation to use installed setuptools
python -m pip install "antlr4-python3-runtime==4.9.3" --no-build-isolation

# 5. Install Robotics Ecosystem (Corrected Versions to match lerobot)
echo "📦 Installing Robotics Ecosystem (Corrected Versions)..."
python -m pip install \
    "lerobot==0.5.0" \
    "mujoco==3.6.0" \
    "gymnasium>=1.1.1" \
    "dm_control==1.0.25" \
    "gym-aloha==0.1.3" \
    "gym-pusht==0.1.5"

# 6. Install Core AI Tools
echo "🧠 Installing AI tools (Corrected Versions)..."
python -m pip install \
    "transformers>=4.48.0" \
    "diffusers>=0.32.0" \
    "accelerate>=1.3.0" \
    "datasets>=3.2.0" \
    "av>=14.0.0" \
    "einops>=0.8.0" \
    "zarr==3.1.6" \
    "draccus>=0.10.0" \
    "omegaconf>=2.3.0" \
    wandb tensorboard tqdm scipy pyyaml

# 7. Environment Variables Configuration
echo "🔧 Configuring Environment..."
if ! grep -q "MUJOCO_GL" ~/.bashrc; then
    {
        echo 'export MUJOCO_GL=egl'
        echo 'export PYOPENGL_PLATFORM=egl'
        echo 'export CUDA_DEVICE_ORDER=PCI_BUS_ID'
        echo 'export TORCH_CUDNN_V8_API_ENABLED=1'
    } >> ~/.bashrc
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "--------------------------------------------------------"
echo "✅ Setup Finalized. Please run: source ~/.bashrc"
echo "--------------------------------------------------------"
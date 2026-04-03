#!/bin/bash

echo "🚀 Starting RTX 5090 Aligned Setup (Headless Rendering Fix)..."

# 1. 系統相依庫 (加入 EGL 與 GL 相關庫)
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglib2.0-0 \
    libegl1-mesa libgles2-mesa libnvidia-compute-535-server \
    ffmpeg libsndfile1 git-lfs ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. 徹底清理
python -m pip uninstall -y xformers opencv-python opencv-contrib-python opencv-python-headless numpy torch torchvision torchaudio

# 3. 安裝 Torch Nightly
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 4. 安裝基礎與模擬環境 (加入 gym-aloha)
echo "📦 Installing Robotics & Gym Envs..."
python -m pip install \
    "numpy>=2.1.0" \
    "lerobot==0.5.0" \
    "mujoco==3.6.0" \
    "gymnasium==1.2.2" \
    "dm_control==1.0.38" \
    "gym-aloha==0.1.3" \
    "gym-pusht==0.1.6"

# 5. 安裝其餘套件
python -m pip install \
    "diffusers==0.35.2" \
    "accelerate==1.12.0" \
    "transformers==5.3.0" \
    "datasets==4.5.0" \
    "opencv-python-headless>=4.11.0.86" \
    "av==15.1.0" \
    "einops==0.8.1" \
    "zarr==3.1.5" \
    "draccus==0.10.0" \
    "omegaconf==2.3.0" \
    wandb tensorboard tqdm scipy pyyaml

# 6. 設定永久環境變數 (針對 RunPod)
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc

echo "------------------------------------------"
echo "✅ Setup Complete. Please run: source ~/.bashrc"
echo "------------------------------------------"
#!/bin/bash

echo "🚀 Starting Final Corrected Setup for H100 (2026-04-04)..."

# 1. 系統相依庫
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglib2.0-0 \
    libegl1-mesa libgles2-mesa \
    ffmpeg libsndfile1 git-lfs ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. 徹底清理
python -m pip uninstall -y xformers opencv-python opencv-contrib-python \
    opencv-python-headless numpy torch torchvision torchaudio lerobot

# 3. 安裝 PyTorch (確認成功的版本)
echo "🔥 Installing PyTorch 2.11.0..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安裝基礎工具 (先裝 OpenCV Headless 以解決依賴問題)
python -m pip install "opencv-python-headless>=4.11.0.86" "numpy>=2.2.0"

# 5. 安裝機器人環境 (修正版本號)
echo "📦 Installing Robotics Ecosystem (Corrected Versions)..."
python -m pip install \
    "lerobot==0.5.0" \
    "mujoco==3.6.0" \
    "gymnasium==1.2.3" \
    "dm_control==1.0.38" \
    "gym-aloha==0.1.5" \
    "gym-pusht==0.1.7"

# 6. 安裝核心 AI 工具 (修正版本號)
echo "🧠 Installing AI tools (Corrected Versions)..."
python -m pip install \
    "transformers==5.5.0" \
    "diffusers==0.36.0" \
    "accelerate==1.13.0" \
    "datasets==4.6.0" \
    "av==15.1.0" \
    "einops==0.8.2" \
    "zarr==3.2.0" \
    "draccus==0.11.0" \
    "omegaconf==2.3.0" \
    wandb tensorboard tqdm scipy pyyaml

# 7. 環境變數設定
echo "🔧 Configuring Environment..."
{
    echo 'export MUJOCO_GL=egl'
    echo 'export PYOPENGL_PLATFORM=egl'
    echo 'export CUDA_DEVICE_ORDER=PCI_BUS_ID'
    echo 'export TORCH_CUDNN_V8_API_ENABLED=1'
} >> ~/.bashrc

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "--------------------------------------------------------"
echo "✅ Setup Finalized. Please run: source ~/.bashrc"
echo "--------------------------------------------------------"
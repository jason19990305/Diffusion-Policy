#!/bin/bash

# =========================================================================
# RunPod H100 Robotics & Vision AI Setup Script (2026-04-04 Updated)
# Target Architecture: NVIDIA Hopper (H100)
# =========================================================================

echo "🚀 Starting H100 Optimized Setup (2026 Spring Edition)..."

# 1. 系統相依庫 (更新驅動介面與 EGL 支持)
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglib2.0-0 \
    libegl1-mesa libgles2-mesa \
    ffmpeg libsndfile1 git-lfs ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. 徹底清理舊版 Python 環境
echo "🧹 Cleaning up legacy packages..."
python -m pip uninstall -y xformers opencv-python opencv-contrib-python \
    opencv-python-headless numpy torch torchvision torchaudio lerobot

# 3. 安裝 PyTorch (2026 最新穩定版 2.11.0 + CUDA 12.8/13.0)
# 對於 H100，強烈建議使用支持 CUDA 12.8+ 的版本以提升 FP8 運算效能
echo "🔥 Installing PyTorch 2.11.0 with CUDA Support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安裝機器人與模擬環境 (更新至 2026/03 最新發布版本)
echo "📦 Installing Robotics & Gym Envs (LeRobot v0.5.0 Ecosystem)..."
python -m pip install \
    "numpy>=2.2.0" \
    "lerobot==0.5.0" \
    "mujoco==3.6.0" \
    "gymnasium==1.2.3" \
    "dm_control==1.0.42" \
    "gym-aloha==0.1.5" \
    "gym-pusht==0.1.7"

# 5. 安裝核心 AI 與 視覺套件 (更新 Transformers v5 系列)
echo "🧠 Installing AI & Computer Vision components..."
python -m pip install \
    "transformers==5.5.0" \
    "diffusers==0.36.0" \
    "accelerate==1.15.0" \
    "datasets==4.6.0" \
    "opencv-python-headless>=4.11.0.86" \
    "av==15.1.0" \
    "einops==0.8.2" \
    "zarr==3.2.0" \
    "draccus==0.11.0" \
    "omegaconf==2.3.0" \
    wandb tensorboard tqdm scipy pyyaml

# 6. 設定 Headless 渲染環境變數 (針對 RunPod H100 優化)
# 使用 EGL 渲染而非傳統 GLX，可大幅提升無頭伺服器的渲染效率
echo "🔧 Configuring Headless Rendering..."
{
    echo 'export MUJOCO_GL=egl'
    echo 'export PYOPENGL_PLATFORM=egl'
    echo 'export CUDA_DEVICE_ORDER=PCI_BUS_ID'
} >> ~/.bashrc

# 7. (選配) H100 效能微調：開啟 Tensor Cores 優化
echo 'export TORCH_CUDNN_V8_API_ENABLED=1' >> ~/.bashrc

echo "------------------------------------------"
echo "✅ Setup Complete. H100 is ready for Robot Learning."
echo "📅 Current Date: 2026-04-04"
echo "👉 Please run: source ~/.bashrc"
echo "------------------------------------------"
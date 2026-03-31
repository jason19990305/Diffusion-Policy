#!/bin/bash

# 1. 更新系統並安裝必要的系統套件 (用於 MuJoCo, OpenCV, 音訊處理)
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    mesa-utils \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 2. 升級 pip
pip install --upgrade pip

# 3. 安裝 PyTorch (根據您的清單使用 cu128 版本)
# 注意：若 RunPod 鏡像已自帶 torch，此步可視情況跳過或調整版本
echo "Installing PyTorch suite..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安裝核心深度學習與 Diffusion 套件
echo "Installing core libraries..."
pip install \
    diffusers==0.35.2 \
    accelerate==1.12.0 \
    transformers==5.3.0 \
    datasets==4.5.0 \
    huggingface_hub==1.7.1

# 5. 安裝機器人環境與 LeRobot
echo "Installing Robotics environments..."
pip install \
    mujoco==3.6.0 \
    gymnasium==1.2.2 \
    dm_control==1.0.38 \
    zarr==3.1.5 \
    lerobot==0.5.0 \
    draccus==0.10.0 \
    omegaconf==2.3.0

# 6. 安裝數據處理與日誌套件
echo "Installing utilities..."
pip install \
    opencv-python==4.9.0.80 \
    av==15.1.0 \
    wandb==0.24.0 \
    tensorboard==2.20.0 \
    tqdm==4.67.1 \
    scipy==1.15.3 \
    einops==0.8.1

# 7. 安裝您的專案以 Editable 模式 (如果有 setup.py 或 pyproject.toml)
# pip install -e .

echo "Installation complete!"
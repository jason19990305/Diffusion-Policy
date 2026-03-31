#!/bin/bash

# 1. 系統環境檢查
echo "Checking Python version..."
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "❌ Error: Python version is $PYTHON_VERSION. Please use a Python 3.12 template."
    # exit 1  # 如果真的不是 3.12 就停止執行
fi

# 2. 安裝系統相依庫 (MuJoCo 與 OpenCV 必備)
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 3. 確保 pip 與路徑正確
python -m pip install --upgrade pip

# 4. 安裝核心套件 (使用 python -m pip 避免路徑錯誤)
echo "Installing core libraries..."
python -m pip install \
    diffusers==0.35.2 \
    accelerate==1.12.0 \
    transformers==5.3.0 \
    datasets==4.5.0 \
    huggingface_hub==1.7.1

# 5. 安裝 LeRobot 0.5.0 與機器人環境
echo "Installing LeRobot 0.5.0 and Robotics stack..."
python -m pip install \
    lerobot==0.5.0 \
    mujoco==3.6.0 \
    gymnasium==1.2.2 \
    dm_control==1.0.38 \
    zarr==3.1.5 \
    draccus==0.10.0 \
    omegaconf==2.3.0

# 6. 安裝其餘工具
echo "Installing utilities..."
python -m pip install \
    opencv-python==4.9.0.80 \
    av==15.1.0 \
    wandb==0.24.0 \
    tensorboard==2.20.0 \
    tqdm==4.67.1 \
    scipy==1.15.3 \
    einops==0.8.1

# 7. 驗證
echo "------------------------------------------"
python -c "import lerobot; print('✅ SUCCESS: LeRobot version', lerobot.__version__, 'is ready!')"
echo "------------------------------------------"
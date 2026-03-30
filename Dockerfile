# --- ALOHA / LeRobot Optimized Dockerfile ---
# Base: PyTorch + CUDA + Devel (to ensure MuJoCo build deps)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# 1. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Persistent directories for RunPod
ENV HF_HOME=/workspace/huggingface
ENV LEROBOT_HOME=/workspace/lerobot

# 2. Install System Dependencies for ALOHA / MuJoCo Rendering
RUN apt-get update && apt-get install -y --no-install-cache \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libosmesa6 \
    libglew2.1 \
    libglfw3 \
    libxml2-dev \
    libjpeg-dev \
    libpng-dev \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Workspace
WORKDIR /workspace

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. [OPTIONAL] Pre-install gym-aloha
# This speeds up pod initialization
RUN pip install --no-cache-dir gym-aloha

# 6. Set Startup Script Target
# We don't COPY the source code here because we will mount / git pull it on RunPod
# for easier iteration (as discussed in implementation plan).

# Default Command: Start a bash shell
CMD ["/bin/bash"]

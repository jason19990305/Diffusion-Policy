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
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libgl1-mesa-glx libosmesa6 libglew-dev libglfw3 \
    libxml2-dev libjpeg-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libavutil-dev libswresample-dev \
    xvfb ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Workspace
WORKDIR /workspace

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 5. Specialized Package Install
# Install lerobot with specific features to handle dependencies better
RUN pip install --no-cache-dir "lerobot[aloha,fe]"

# 6. Copy Source Code (Uses .dockerignore to skip data)
COPY . .
RUN chmod +x scripts/*.sh

# Default Command: Start a bash shell
CMD ["/bin/bash"]

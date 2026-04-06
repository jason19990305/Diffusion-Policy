#!/bin/bash

# ==========================================
# 🚀 H100 High-Speed Training Config (Optimized for 94GB VRAM)
# ==========================================

# 1. Core Parameter Tuning
# Recommendation: ALOHA Sim can handle BS 512 or even 1024 on H100
BS=256 
LR=2e-4        # Higher BS may require slightly higher LR (e.g., 8e-4 to 1e-3)
STEPS=100000     # Total steps target for convergence
SAVE=5000
WORKERS=16      # Use more workers for high-core CPUs to avoid I/O bottlenecks

# 2. Environment Variable Optimization (Hopper Architecture)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDNN_V8_API_ENABLED=1
# Reduce VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

echo "----------------------------------------------------"
echo "🔥 H100 NVIDIA Hopper Mode Activated"
echo "📦 Batch Size: $BS"
echo "⚙️  Learning Rate: $LR"
echo "🧵 Num Workers: $WORKERS"
echo "----------------------------------------------------"

# Ensure checkpoint directory exists
mkdir -p checkpoints

# 3. Execute Training
# Use stdbuf to ensure real-time logging to console
stdbuf -oL python aloha_train.py \
    --batch_size "$BS" \
    --lr "$LR" \
    --total_steps "$STEPS" \
    --num_workers "$WORKERS" \
    --save_interval "$SAVE" \

echo "✅ Training task completed"
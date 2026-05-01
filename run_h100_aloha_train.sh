#!/bin/bash

# ==========================================
# 🛠️ tmux Session Management (SSH Protect)
# ==========================================
SESSION_NAME="aloha_train"

if [ -z "$TMUX" ]; then
    # Check if session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "🚀 Starting training in new tmux session: $SESSION_NAME"
        # Start session and run the same script inside it
        # 'read' allows the session to stay open after the script ends or crashes
        tmux new-session -d -s "$SESSION_NAME" "bash $0; read"
        echo "✅ Training launched in background."
        echo "👉 Command to attach: tmux attach -t $SESSION_NAME"
        exit 0
    else
        echo "⚠️  Session '$SESSION_NAME' is already running."
        echo "👉 Command to attach: tmux attach -t $SESSION_NAME"
        echo "👉 Command to kill:   tmux kill-session -t $SESSION_NAME"
        exit 0
    fi
fi

# ==========================================
# 🚀 H100 High-Speed Training Config (Optimized for 94GB VRAM)
# ==========================================

# 1. Core Parameter Tuning
# Recommendation: ALOHA Sim can handle BS 512 or even 1024 on H100
BS=64 
LR=2e-4        # Optimal LR for 5090 with Augmentation
STEPS=600000        # [Update] Shortened training steps to 600K to allow learning rate (Cosine) to decay earlier, preventing oscillation and helping stable convergence.
SAVE=10000       # Less frequent saving
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
# And use 'tee' to save a persistent log file
LOG_FILE="checkpoints/train_$(date +%Y%m%d_%H%M%S).log"
stdbuf -oL python aloha_train.py \
    --batch_size "$BS" \
    --lr "$LR" \
    --total_steps "$STEPS" \
    --num_workers "$WORKERS" \
    --save_interval "$SAVE" \
    2>&1 | tee "$LOG_FILE"

echo "✅ Training task completed"
#!/bin/bash

# =================================================================
# H100 Optimized Training Script with Flexible Steps
# =================================================================

# --- 核心參數區 ---
# 針對 H100 80GB 的最佳配置
BATCH_SIZE=256
LEARNING_RATE=8e-4  # 配合大 Batch Size 調高 LR
NUM_WORKERS=8       # 配合 RunPod vCPU 數量

# --- 指定 Steps (這裡可以依需求改 50000 或 80000) ---
TOTAL_STEPS=50000
SAVE_INTERVAL=5000  # 每 5000 步存一次檔
# -----------------

echo "----------------------------------------------------"
echo "🚀 H100 DEEP LEARNING MODE"
echo "📦 Batch Size: $BATCH_SIZE"
echo "🎯 Total Steps: $TOTAL_STEPS (Approx $(($TOTAL_STEPS / 100)) Epochs)"
echo "----------------------------------------------------"

# 啟動訓練
python aloha_train.py \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --total_steps $TOTAL_STEPS \
    --num_workers $NUM_WORKERS

echo "✅ Training complete! Check /checkpoints for your model."
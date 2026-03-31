#!/bin/bash

# =================================================================
# H100 Optimized Training Script for Aloha Diffusion Policy
# =================================================================

# 1. 環境變數設定
# 增加卷積算子搜尋優化，讓 H100 跑得更快
export TORCH_CUDNN_V8_API_ENABLED=1
export WANDB_MODE=online  # 如果沒裝 wandb 可以改成 offline

# 2. 超參數設定 (針對 H100 80GB)
# ---------------------------------------------------------
# 原本 BS=32, 現在提高到 256 (H100 處理大 Batch 極快)
BATCH_SIZE=256

# 隨著 Batch Size 增加，LR 也要適度調高以維持收斂速度
# 遵循 Linear Scaling Rule: 2e-4 * (256/32) = 1.6e-3，但保險起見設為 8e-4
LEARNING_RATE=8e-4

# 設定總步數 (50,000 步在 BS=256 下代表看過更多數據)
TOTAL_STEPS=50000

# 設定 CPU 讀取線程 (RunPod H100 建議設為 8)
NUM_WORKERS=8
# ---------------------------------------------------------

echo "🚀 Starting H100 Optimized Training..."
echo "📦 Batch Size: $BATCH_SIZE"
echo "💡 Learning Rate: $LEARNING_RATE"
echo "🧵 CPU Workers: $NUM_WORKERS"

python aloha_train.py \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --total_steps $TOTAL_STEPS \
    --num_workers $NUM_WORKERS

echo "✅ Training session finished."
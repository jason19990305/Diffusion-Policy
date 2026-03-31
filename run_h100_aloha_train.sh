#!/bin/bash

# --- 參數設定 ---
BS=256
LR=8e-4
STEPS=50000
WORKERS=8
SAVE=5000

echo "----------------------------------------------------"
echo "🚀 H100 DEEP LEARNING MODE"
echo "📦 Batch Size: $BS"
echo "----------------------------------------------------"

# 確保存檔目錄存在
mkdir -p checkpoints

# 執行指令 (寫成單行最保險，避免換行連接符出錯)
python aloha_train.py --batch_size "$BS" --lr "$LR" --total_steps "$STEPS" --num_workers "$WORKERS" --save_interval "$SAVE"
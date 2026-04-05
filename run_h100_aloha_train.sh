#!/bin/bash

# ==========================================
# 🚀 H100 極速訓練配置 (Optimized for 94GB VRAM)
# ==========================================

# 1. 核心參數調整
# 建議：H100 跑 ALOHA Sim 建議可以衝到 512 甚至 1024
BS=512 
LR=2e-4        # 隨著 BS 增加，LR 也稍微調高 (原 8e-4 -> 1e-3)
STEPS=12500     # 因為 BS 變大，總 Steps 可以適度減少，節省時間
SAVE=5000
WORKERS=16      # H100 通常配備高核心 CPU，增加 Worker 避免 I/O 瓶頸

# 2. 環境變數優化 (針對 NVIDIA Hopper 架構)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDNN_V8_API_ENABLED=1
# 減少顯存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

echo "----------------------------------------------------"
echo "🔥 H100 NVIDIA Hopper Mode Activated"
echo "📦 Batch Size: $BS"
echo "⚙️  Learning Rate: $LR"
echo "🧵 Num Workers: $WORKERS"
echo "----------------------------------------------------"

# 確保存檔目錄存在
mkdir -p checkpoints

# 3. 執行訓練
# 加上 stdbuf 確保 Log 即時輸出到控制台
stdbuf -oL python aloha_train.py \
    --batch_size "$BS" \
    --lr "$LR" \
    --total_steps "$STEPS" \
    --num_workers "$WORKERS" \
    --save_interval "$SAVE" \

echo "✅ 訓練任務已完成"
#!/bin/bash

# =================================================================
# H100 專用 Diffusion Policy 訓練腳本 (Aloha 任務)
# =================================================================

# 1. 確保環境變數正確 (預防多 GPU 環境出錯)
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="aloha_diffusion_h100"

# 2. 啟動訓練
# 注意：這裡的參數是針對 H100 80GB 優化的超參數
python aloha_train.py \
    --dataset_repo_id "jason19990305/aloha_sim_transfer_cube_human" \
    --batch_size 256 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --num_epochs 1000 \
    --device "cuda" \
    --use_amp True \
    --seed 42 \
    --eval_freq 50 \
    --save_freq 100 \
    --checkpoint_path "./checkpoints/h100_optimized"

# 說明：
# --batch_size 256: H100 80GB 綽綽有餘。增大 Batch 可以讓梯度更穩定並大幅縮短訓練時間。
# --use_amp True: 啟用自動混合精度 (Automatic Mixed Precision)，H100 跑 BF16/FP16 非常快。
# --num_workers 8: 根據您 RunPod 截圖顯示只有 8 vCPU，這裡設為 8 是為了不讓 CPU 成為數據讀取的瓶頸。
# --learning_rate 1e-4: 隨著 Batch Size 增大，我們微調了學習率以維持收斂速度。
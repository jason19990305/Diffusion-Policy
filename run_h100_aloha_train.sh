#!/bin/bash
set -e  # 如果任何指令出錯，立刻停止腳本

# ... 前面的參數設定 ...

echo "🚀 Starting H100 Optimized Training..."

# 執行訓練
python aloha_train.py \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --total_steps $TOTAL_STEPS \
    --num_workers $NUM_WORKERS \
    --save_interval $SAVE_INTERVAL

# 只有上面成功才會執行到這裡
echo "✅ Training session finished successfully."
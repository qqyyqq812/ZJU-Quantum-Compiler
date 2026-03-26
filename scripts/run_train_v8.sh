#!/bin/bash
# V8 训练启动脚本
# 基于深度反思的三大改进：
#   1. SWAP 惩罚从 -0.5 → -2.0（主导奖励信号）
#   2. 门执行奖励从 1.0 → 0.3（降低噪声）
#   3. 随机初始映射（每 episode 随机映射，适应多种情况）
#
# 用法:
#   bash run_train_v8.sh              # 全新训练
#   bash run_train_v8.sh resume       # 从 V7.2 best 恢复

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SAVE_DIR="models/v8_rewarded"
LOG_FILE="training_v8.log"

cd ~/projects/量子电路
source .venv/bin/activate

if [ "$1" = "resume" ]; then
    echo "🔄 从 V7.2 best 模型微调..."
    RESUME_FLAG="--resume models/v7_final_v2/v7_linear_5_best.pt"
else
    echo "🆕 全新 V8 训练..."
    RESUME_FLAG=""
fi

echo "📋 V8 训练配置:"
echo "   SWAP 惩罚: -1.0 (温和退火，防死锁)"
echo "   门执行奖励: 1.0"
echo "   完成奖励: 20.0"
echo "   距离缩减: 0.5"
echo "   随机映射: 启用"
echo "   保存到: $SAVE_DIR"
echo ""

nohup python -u -m src.compiler.train \
    --topology linear_5 \
    --episodes 30000 \
    --curriculum \
    --save-dir "$SAVE_DIR" \
    --reward-gate 1.0 \
    --penalty-swap -1.0 \
    --reward-done 20.0 \
    --distance-coef 0.5 \
    --random-mapping \
    $RESUME_FLAG \
    > "$LOG_FILE" 2>&1 &

echo "✅ V8 训练已启动 (PID: $!)"
echo "   日志: tail -f $LOG_FILE"

#!/bin/bash
# V9 训练启动脚本 — IBM Tokyo 20Q 拓扑
# 课程学习自动升阶: 3Q warm-up → 5Q → 10Q → 20Q master
#
# 用法:
#   bash scripts/run_train_v9_20Q.sh         # 全新训练
#   bash scripts/run_train_v9_20Q.sh resume  # 从最新 checkpoint 恢复

set -e

cd ~/projects/量子电路
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_DYNAMO_DISABLE=1

SAVE_DIR="models/v9_tokyo20"
LOG_FILE="logs/training_v9_20Q.log"

# 找到最新 checkpoint
LATEST_CKPT=$(ls -t ${SAVE_DIR}/checkpoint_ep*.pt 2>/dev/null | head -1)

if [ "$1" = "resume" ] && [ -n "$LATEST_CKPT" ]; then
    echo "🔄 从最新 checkpoint 恢复: $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "🆕 全新 V9 训练..."
    RESUME_FLAG=""
fi

echo "📋 V9 训练配置 (IBM Tokyo 20Q):"
echo "   拓扑: ibm_tokyo (20Q, 43 edges, diameter 4)"
echo "   SWAP 惩罚: -1.0"
echo "   门执行奖励: 1.0"
echo "   完成奖励: 20.0"
echo "   距离缩减: 0.5"
echo "   随机映射: 启用"
echo "   课程学习: 3Q→5Q→10Q→20Q"
echo "   保存到: $SAVE_DIR"
echo ""

nohup python -u -m src.compiler.train \
    --topology ibm_tokyo \
    --qubits 20 \
    --episodes 50000 \
    --curriculum \
    --save-dir "$SAVE_DIR" \
    --reward-gate 1.0 \
    --penalty-swap -1.0 \
    --reward-done 20.0 \
    --distance-coef 0.5 \
    --random-mapping \
    $RESUME_FLAG \
    > "$LOG_FILE" 2>&1 &

echo "✅ V9 训练已启动 (PID: $!)"
echo "   日志: tail -f $LOG_FILE"

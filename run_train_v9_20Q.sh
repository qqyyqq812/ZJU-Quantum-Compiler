#!/bin/bash
# V9 20Q IBM Tokyo 训练启动脚本
# 用法:
#   bash run_train_v9_20Q.sh          # 从头训练
#   bash run_train_v9_20Q.sh resume   # 从 checkpoint 恢复

set -euo pipefail

# ── 项目根目录 & venv ──
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# ── 环境变量 (防止 PyTorch 多线程争用) ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_DYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1        # 防止 nohup 下日志块缓存导致 0 字节
export PYTHONPATH="${PROJECT_DIR}"

# ── 训练参数 ──
TOPOLOGY="ibm_tokyo"
EPISODES=50000
SAVE_DIR="models/v10_tokyo20"

# 自动获取最大 Episode 数字的 checkpoint
CHECKPOINT=$(ls -v "${SAVE_DIR}"/checkpoint_ep*.pt 2>/dev/null | tail -n 1 || echo "")

echo "🚀 V9 20Q IBM Tokyo 训练"
echo "   拓扑: ${TOPOLOGY}"
echo "   Episodes: ${EPISODES}"
echo "   保存目录: ${SAVE_DIR}"

# ── 判断恢复模式 ──
RESUME_ARG=""
if [ "${1:-}" = "resume" ] && [ -n "${CHECKPOINT}" ] && [ -f "${CHECKPOINT}" ]; then
    RESUME_ARG="--resume ${CHECKPOINT}"
    echo "   🔄 从最新 checkpoint 恢复: ${CHECKPOINT}"
else
    echo "   🆕 从头开始训练 (或未找到 checkpoint)"
fi

# ── 启动训练 ──
python -m src.compiler.train \
    --topology "${TOPOLOGY}" \
    --qubits 20 \
    --episodes "${EPISODES}" \
    --rollout-steps 256 \
    --save-dir "${SAVE_DIR}" \
    --curriculum \
    --lr 3e-4 \
    --eval-interval 1000 \
    --reward-gate 1.0 \
    --penalty-swap -1.0 \
    --reward-done 20.0 \
    --distance-coef 0.5 \
    --random-mapping \
    --soft-mask \
    --tabu-size 4 \
    --checkpoint-interval 500 \
    ${RESUME_ARG} \
    2>&1 | tee "${SAVE_DIR}/train_v9_$(date +%Y%m%d_%H%M).log"

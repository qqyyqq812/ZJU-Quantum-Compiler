#!/bin/bash
# ============================================================
# V14 量子电路 AI 路由器训练脚本 — RTX 5090 GPU
# ============================================================
# 使用方法 (参见 .claude/rules/deployment.md):
#   1. 本地 git push 最新代码
#   2. SSH 到 AutoDL GPU 服务器
#   3. cd /root/quantum && git pull
#   4. bash run_train_v14.sh
# ============================================================
set -euo pipefail

CONFIG="${1:-configs/v14_baseline.yaml}"

echo "=========================================="
echo "  V14 量子路由器训练"
echo "  配置文件: $CONFIG"
echo "  (见 docs/technical/decisions.md §V14)"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU"

# 读取 save_dir 从 yaml (简单 grep, 够用)
SAVE_DIR=$(grep "^  save_dir:" "$CONFIG" | awk '{print $2}' || echo "models/v14_tokyo20")
mkdir -p "$SAVE_DIR"

python -u -m src.compiler.train \
    --config "$CONFIG" \
    2>&1 | tee "$SAVE_DIR/training_v14.log"

echo ""
echo "V14 训练完成"
echo "模型保存在: $SAVE_DIR"
echo "日志: $SAVE_DIR/training_v14.log"
echo "下一步: 生成 eval_report_v14.md"

#!/bin/bash
# ============================================================
# V13 量子电路 AI 路由器训练脚本 — RTX 5090 GPU
# ============================================================
# 用法:
#   1. 将代码推送到 GitHub: git push
#   2. SSH 到 GPU 服务器: ssh -p 14191 root@connect.westd.seetacloud.com
#   3. 在 GPU 服务器上:
#        cd /root && git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git quantum
#        cd quantum && pip install -r requirements.txt
#        bash run_train_v13.sh
# ============================================================
set -euo pipefail

echo "=========================================="
echo "  V13 量子路由器训练 — SABRE 相对奖励"
echo "  RTX 5090 (32GB) 全力输出"
echo "=========================================="

# 检测 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: No GPU detected"

# 训练参数 (V13 默认已内置最优配置)
python -m src.compiler.train \
    --topology ibm_tokyo \
    --qubits 20 \
    --episodes 100000 \
    --rollout-steps 32768 \
    --save-dir models/v13_tokyo20 \
    --curriculum \
    --lr 3e-4 \
    --eval-interval 500 \
    --checkpoint-interval 2000 \
    2>&1 | tee models/v13_tokyo20/training_v13.log

echo ""
echo "✅ V13 训练完成!"
echo "模型保存在: models/v13_tokyo20/"
echo "日志文件: models/v13_tokyo20/training_v13.log"

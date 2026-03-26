#!/bin/bash
# V7 50K 训练启动脚本
cd /home/qq/projects/量子电路
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

python -u -m src.compiler.train \
    --topology linear_5 \
    --episodes 50000 \
    --curriculum \
    --save-dir models/v7_full \
    --eval-interval 5000 \
    2>&1 | tee training_50k_v2.log

echo "训练结束，exit code: $?"

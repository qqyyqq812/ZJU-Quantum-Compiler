#!/bin/bash
export PYTHONPATH=/home/qq/projects/量子电路
cd /home/qq/projects/量子电路
echo "Running AI..."
.venv/bin/python -u scripts/eval_v12_ai.py > results/v12_tokyo20/log_ai.txt 2>&1
echo "Running SABRE..."
.venv/bin/python -u scripts/eval_v12_sabre.py > results/v12_tokyo20/log_sabre.txt 2>&1
echo "Plotting..."
.venv/bin/python -u scripts/eval_v12_plot.py > results/v12_tokyo20/log_plot.txt 2>&1
echo "DONE"

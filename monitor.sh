#!/bin/bash
# 量子电路训练监控脚本 — 在 GPU 服务器上运行
# 用法: bash monitor.sh [interval_sec]
INTERVAL=${1:-10}
LOG=/root/quantum/models/v14_tokyo20/training_v14.log
while true; do
    clear
    echo "===== 量子电路 V14 训练监控 ===== $(date '+%H:%M:%S')"
    echo ""
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null
    echo ""
    echo "--- CPU / MEM ---"
    echo "Workers: $(ps aux | grep 'src.compiler.train' | grep -v grep | wc -l)"
    free -h | awk 'NR==2{printf "RAM: %s used / %s total\n", $3, $2}'
    echo ""
    echo "--- 最新训练进度 ---"
    tail -8 "$LOG" 2>/dev/null || echo "No log yet"
    echo ""
    echo "--- SABRE 缓存 (SWAP 历史最近10条) ---"
    python -c "
import json, os
f = '/root/quantum/models/v14_tokyo20/history_v7_ibm_tokyo.json'
if os.path.exists(f):
    h = json.load(open(f))
    es = h.get('episode_swaps', [])[-10:]
    cs = h.get('curriculum_stages', [])
    stage = cs[-1] if cs else '?'
    print(f'Total episodes: {len(h.get(\"episode_swaps\",[]))} | Stage: {stage}')
    print(f'Last 10 SWAPs: {[round(x,1) for x in es]}')
    eval_s = h.get('eval_swaps', [])
    eval_sabre = h.get('eval_sabre_swaps', [])
    if eval_s:
        print(f'Last eval: AI={round(eval_s[-1],1)} vs SABRE={round(eval_sabre[-1],1) if eval_sabre else \"?\"}')
else:
    print('No history yet')
" 2>/dev/null
    echo ""
    echo "(刷新间隔 ${INTERVAL}s — Ctrl+C 退出)"
    sleep "$INTERVAL"
done

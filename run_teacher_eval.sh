#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  量子电路 AI 编译器 — 一键老师验收评测脚本
# ═══════════════════════════════════════════════════════════════
#  用法:
#    bash run_teacher_eval.sh                    # 使用默认最佳模型
#    bash run_teacher_eval.sh <model_path>       # 指定模型文件
#
#  功能:
#    1. 自动检测并激活 Python 虚拟环境
#    2. 锁死多线程环境变量（防止 CPU 死锁）
#    3. 运行 SABRE vs AI 全量对比评测
#    4. 生成 IBM Tokyo 20Q 拓扑可视化图
#    5. 生成路由动态 GIF 动画
#    6. 彩色输出美化的对比结果表格
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── 颜色定义 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── 项目根目录 ──
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 环境保护 (防止 CPU 死锁) ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_DYNAMO_DISABLE=1
export PYTHONPATH="${PROJECT_DIR}"

# ── 自动激活虚拟环境 ──
if [ -d "${PROJECT_DIR}/.venv" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
    echo -e "${GREEN}✓${NC} Python 虚拟环境已激活: ${PROJECT_DIR}/.venv"
else
    echo -e "${RED}✗ 未找到虚拟环境! 请先运行: python -m venv .venv && pip install -r requirements.txt${NC}"
    exit 1
fi

# ── 模型路径 ──
DEFAULT_MODEL="${PROJECT_DIR}/models/v9_tokyo20/v7_ibm_tokyo_best.pt"
MODEL_PATH="${1:-${DEFAULT_MODEL}}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}✗ 模型文件不存在: ${MODEL_PATH}${NC}"
    exit 1
fi

# ── 输出目录 ──
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

echo ""
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  量子电路 AI 编译器 — 验收评测${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}项目目录${NC}: ${PROJECT_DIR}"
echo -e "  ${BLUE}模型文件${NC}: ${MODEL_PATH}"
echo -e "  ${BLUE}目标拓扑${NC}: IBM Tokyo (20 Qubits)"
echo -e "  ${BLUE}对比基线${NC}: Qiskit SABRE (optimization_level=3)"
echo ""

# ═══════════════════════════════════════════════════════════
# Phase 1: SABRE vs AI 全量对比评测
# ═══════════════════════════════════════════════════════════
echo -e "${BOLD}${YELLOW}▶ Phase 1: SABRE vs AI 性能对比${NC}"
echo ""

python -c "
import time, sys
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from src.benchmarks.topologies import get_topology
from src.benchmarks.circuits import generate_random, generate_qft, generate_qaoa, generate_grover
from src.compiler.inference_v8 import load_policy, compile_multi_trial

cm = get_topology('ibm_tokyo')
policy, _ = load_policy('${MODEL_PATH}', 'ibm_tokyo')

circuits = [
    ('QFT-5',      generate_qft(5)),
    ('QAOA-5',     generate_qaoa(5, p=1)),
    ('Grover-5',   generate_grover(5)),
    ('Random-5d4', generate_random(5, depth=4, seed=42)),
    ('Random-5d6', generate_random(5, depth=6, seed=42)),
    ('QFT-10',     generate_qft(10)),
    ('QAOA-10',    generate_qaoa(10, p=1)),
    ('Random-10d6',generate_random(10, depth=6, seed=42)),
]

def sabre(qc):
    pm = generate_preset_pass_manager(optimization_level=3, coupling_map=cm, routing_method='sabre')
    t = pm.run(qc)
    return dict(t.count_ops()).get('swap', 0)

# 表头
print('\033[1m' + f\"{'Circuit':14s} | {'CX':>4s} | {'SABRE':>6s} | {'AI(MT×20)':>10s} | {'Winner':>8s}\" + '\033[0m')
print('-' * 55)

total_sabre, total_ai, ai_wins = 0, 0, 0
for name, qc in circuits:
    cx = dict(qc.count_ops()).get('cx', 0)
    s = sabre(qc)
    t0 = time.time()
    r = compile_multi_trial(qc, policy, cm, n_trials=20, soft_mask=True, tabu_size=4)
    dt = time.time() - t0
    a = r.swaps

    if a <= s:
        winner = '\033[32m✓ AI\033[0m'
        ai_wins += 1
    else:
        winner = '\033[31m✗ SABRE\033[0m'

    total_sabre += s
    total_ai += a
    print(f'{name:14s} | {cx:4d} | {s:6d} | {a:7d} ({dt:.1f}s) | {winner}')

print('-' * 55)
ratio = total_sabre / total_ai if total_ai > 0 else 0
color = '\033[32m' if total_ai <= total_sabre else '\033[31m'
print(f\"{'TOTAL':14s} |      | {total_sabre:6d} | {total_ai:10d} | {color}{ratio:.2f}x\033[0m\")
print()
print(f'AI 胜出比例: {ai_wins}/{len(circuits)} ({ai_wins/len(circuits):.0%})')
"

echo ""

# ═══════════════════════════════════════════════════════════
# Phase 2: 拓扑可视化
# ═══════════════════════════════════════════════════════════
echo -e "${BOLD}${YELLOW}▶ Phase 2: IBM Tokyo 拓扑可视化${NC}"
echo ""

python -c "
from src.benchmarks.topologies import get_topology
from src.visualization.topology_visualizer import render_topology

cm = get_topology('ibm_tokyo')
render_topology(cm, topology_name='ibm_tokyo', save_path='${RESULTS_DIR}/ibm_tokyo_topology.png')
"

echo ""

# ═══════════════════════════════════════════════════════════
# Phase 3: 路由动画 GIF
# ═══════════════════════════════════════════════════════════
echo -e "${BOLD}${YELLOW}▶ Phase 3: 路由可视化动画${NC}"
echo ""

python -c "
from src.benchmarks.topologies import get_topology
from src.benchmarks.circuits import generate_qft
from src.compiler.inference_v8 import load_policy
from src.visualization.topology_visualizer import run_captured_route, generate_route_gif

cm = get_topology('ibm_tokyo')
policy, _ = load_policy('${MODEL_PATH}', 'ibm_tokyo')
circuit = generate_qft(5)

print('  捕获 AI 路由过程...')
capturer, total_swaps, completed = run_captured_route(circuit, policy, cm, 'ibm_tokyo')
print(f'  路由完成: SWAP={total_swaps}, 成功={completed}')

generate_route_gif(capturer, cm, topology_name='ibm_tokyo',
                   output_path='${RESULTS_DIR}/route_animation.gif',
                   max_frames=40, frame_duration=400)
"

echo ""

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ 验收评测完成!${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}生成文件:${NC}"
echo -e "    📊 拓扑图:   ${RESULTS_DIR}/ibm_tokyo_topology.png"
echo -e "    🎬 路由动画: ${RESULTS_DIR}/route_animation.gif"
echo ""
echo -e "  ${YELLOW}提示: 上述文件可直接嵌入论文或展示 PPT${NC}"
echo ""

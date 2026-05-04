#!/bin/bash
# 量子电路 AI 编译器 — 老师展示验收脚本
#
# 这个脚本生成一个可直接展示的本地证据包：
#   1. qcompiler info
#   2. 一个小 QASM 电路的 SABRE 编译结果
#   3. qcompiler 的 SABRE/AI 小表格
#   4. MQT-Bench 5Q 演示报告
#   5. IBM Tokyo 拓扑图
#
# 注意：当前 V14/V15 checkpoint 尚未超过 SABRE。本脚本会诚实展示
# AI status = OK / INCOMPLETE / N/A，不把未完成路由包装成胜利。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results/teacher_demo"
MODEL_PATH="${1:-${PROJECT_DIR}/models/v14_tokyo20/checkpoint_ep25333.pt}"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_DYNAMO_DISABLE=1
export PYTHONPATH="${PROJECT_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "[error] 未找到虚拟环境 Python: ${PYTHON}"
    echo "        请先在项目根目录准备 .venv 并安装依赖。"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[warn] AI 模型不存在: ${MODEL_PATH}"
    echo "       后续 AI 列会显示 N/A；SABRE 评测仍可用。"
fi

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  ZJU Quantum Compiler — teacher demo evidence pack"
echo "============================================================"
echo "Project: ${PROJECT_DIR}"
echo "Output:  ${RESULTS_DIR}"
echo "Model:   ${MODEL_PATH}"
echo

echo "[1/5] qcompiler info"
"${PYTHON}" -m src.cli info | tee "${RESULTS_DIR}/01_qcompiler_info.txt"

echo
echo "[2/5] 生成小型 QASM 并用 SABRE 编译"
"${PYTHON}" - <<PY
from pathlib import Path
from qiskit import QuantumCircuit, qasm2

out = Path("${RESULTS_DIR}")
qc = QuantumCircuit(5, name="teacher_qft_like_5")
qc.h(0)
qc.cx(0, 4)
qc.cx(1, 3)
qc.cx(2, 4)
qasm2.dump(qc, out / "teacher_qft_like_5.qasm")
print(out / "teacher_qft_like_5.qasm")
PY
"${PYTHON}" -m src.cli compile \
    "${RESULTS_DIR}/teacher_qft_like_5.qasm" \
    --topology tokyo \
    --backend sabre \
    --output "${RESULTS_DIR}/teacher_qft_like_5_sabre.qasm" \
    | tee "${RESULTS_DIR}/02_sabre_compile.txt"

echo
echo "[3/5] qcompiler 小规模 SABRE/AI 对比"
"${PYTHON}" -m src.cli eval \
    --circuits qft_5,qaoa_5,ghz_5 \
    --topology tokyo \
    --model "${MODEL_PATH}" \
    --max-steps 600 \
    | tee "${RESULTS_DIR}/03_qcompiler_eval.txt"

echo
echo "[4/5] MQT-Bench 5Q 演示报告"
"${PYTHON}" scripts/eval_mqt_bench.py \
    --ai-model "${MODEL_PATH}" \
    --n-qubits 5 \
    --benchmarks qft,qaoa,ghz,vqe \
    --max-steps 600 \
    --output "${RESULTS_DIR}/04_mqt_5q_demo.md" \
    | tee "${RESULTS_DIR}/04_mqt_5q_demo.log"

echo
echo "[5/5] IBM Tokyo 拓扑图"
"${PYTHON}" - <<PY
from pathlib import Path
from src.benchmarks.topologies import get_topology
from src.visualization.topology_visualizer import render_topology

out = Path("${RESULTS_DIR}") / "05_ibm_tokyo_topology.png"
cm = get_topology("ibm_tokyo")
render_topology(cm, topology_name="ibm_tokyo", save_path=str(out))
print(out)
PY

cat > "${RESULTS_DIR}/README.md" <<EOF
# Teacher Demo Evidence Pack

生成时间：$(date '+%Y-%m-%d %H:%M:%S')

本目录是量子电路 AI 编译项目的现场展示证据包。当前口径：

- SABRE 是稳定基线，可现场复现。
- V14 checkpoint 可以加载并运行 AI Router，但 P1 结果尚未超过 SABRE。
- AI status 会明确显示 OK / INCOMPLETE / N/A。
- 项目闭环重点是问题建模、工程实现、真实评测、失败诊断和 V15 后续路线。

文件：

- \`01_qcompiler_info.txt\`：CLI、模型和拓扑状态。
- \`02_sabre_compile.txt\`：小 QASM 电路的 SABRE 编译输出。
- \`03_qcompiler_eval.txt\`：CLI 小规模 SABRE/AI 对比。
- \`04_mqt_5q_demo.md\`：MQT-Bench 5Q 表格报告。
- \`04_mqt_5q_demo.json\`：MQT-Bench 机读结果。
- \`05_ibm_tokyo_topology.png\`：IBM Tokyo 20Q 拓扑图。
EOF

echo
echo "============================================================"
echo "Done. Evidence pack generated at:"
echo "  ${RESULTS_DIR}"
echo "============================================================"

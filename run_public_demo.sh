#!/bin/bash
# ZJU Quantum Compiler — public local demo script
#
# 这个脚本生成一个公开项目本地演示包：
#   1. qcompiler info
#   2. 公开 examples 中 QFT5 的 SABRE 编译结果
#   3. qcompiler 的 SABRE/AI 小表格
#   4. MQT-Bench 5Q 演示报告
#   5. IBM Tokyo 拓扑图
#   6. SabreSwap heuristic lab 证据
#
# 注意：当前 V14/V15 checkpoint 尚未超过 SABRE。本脚本会诚实展示
# AI status = OK / INCOMPLETE / N/A，不把未完成路由包装成胜利。
# 当前公开网站主线是 SabreSwap / LightSABRE heuristic lab。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results/public_demo"
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
echo "  ZJU Quantum Compiler — public local demo pack"
echo "============================================================"
echo "Project: ${PROJECT_DIR}"
echo "Output:  ${RESULTS_DIR}"
echo "Model:   ${MODEL_PATH}"
echo

echo "[1/6] qcompiler info"
"${PYTHON}" -m src.cli info | tee "${RESULTS_DIR}/01_qcompiler_info.txt"

echo
echo "[2/6] 使用公开 examples/qft5.qasm 做 SABRE 编译"
"${PYTHON}" -m src.cli compile \
    "${PROJECT_DIR}/examples/qft5.qasm" \
    --topology tokyo \
    --backend sabre \
    --heuristic lookahead \
    --output "${RESULTS_DIR}/qft5_sabre.qasm" \
    | tee "${RESULTS_DIR}/02_sabre_compile.txt"

echo
echo "[3/6] qcompiler 小规模 SABRE/AI 对比"
"${PYTHON}" -m src.cli eval \
    --circuits qft_5,qaoa_5,ghz_5 \
    --topology tokyo \
    --model "${MODEL_PATH}" \
    --max-steps 600 \
    | tee "${RESULTS_DIR}/03_qcompiler_eval.txt"

echo
echo "[4/6] MQT-Bench 5Q 演示报告"
"${PYTHON}" scripts/eval_mqt_bench.py \
    --ai-model "${MODEL_PATH}" \
    --n-qubits 5 \
    --benchmarks qft,qaoa,ghz,vqe \
    --max-steps 600 \
    --output "${RESULTS_DIR}/04_mqt_5q_demo.md" \
    | tee "${RESULTS_DIR}/04_mqt_5q_demo.log"

echo
echo "[5/6] IBM Tokyo 拓扑图"
"${PYTHON}" - <<PY
from pathlib import Path
from src.benchmarks.topologies import get_topology
from src.visualization.topology_visualizer import render_topology

out = Path("${RESULTS_DIR}") / "05_ibm_tokyo_topology.png"
cm = get_topology("ibm_tokyo")
render_topology(cm, topology_name="ibm_tokyo", save_path=str(out))
print(out)
PY

echo
echo "[6/6] SabreSwap heuristic lab 证据"
"${PYTHON}" scripts/experiment_sabre_trials.py \
    --examples qft5 ghz5 qaoa5 qft10 qaoa10 ghz10 vqe10 \
    --trials 1 \
    | tee "${RESULTS_DIR}/06_sabre_heuristic_lab.md"

"${PYTHON}" scripts/experiment_bounded_search.py \
    --examples qft10 \
    --branch-factor 2 \
    --max-depth 2 \
    --max-steps 400 \
    | tee "${RESULTS_DIR}/07_bounded_search_negative_sample.md"

cat > "${RESULTS_DIR}/README.md" <<EOF
# Public Local Demo Pack

生成时间：$(date '+%Y-%m-%d %H:%M:%S')

本目录是 ZJU Quantum Compiler 的本地演示证据包。当前口径：

- SABRE / LightSABRE 是稳定实用基线，可现场复现。
- 网站主线是 heuristic lab：比较 basic / lookahead / decay。
- V14 checkpoint 可以加载并运行 AI Router，但 P1 结果尚未超过 SABRE。
- AI status 会明确显示 OK / INCOMPLETE / N/A。
- bounded search v1 是负样本，不进入网站默认算法。
- 项目闭环重点是公开网页、示例电路、CLI/API、真实评测、失败复盘和后续路线。

文件：

- \`01_qcompiler_info.txt\`：CLI、模型和拓扑状态。
- \`02_sabre_compile.txt\`：公开 QFT5 示例的 SABRE 编译输出。
- \`03_qcompiler_eval.txt\`：CLI 小规模 SABRE/AI 对比。
- \`04_mqt_5q_demo.md\`：MQT-Bench 5Q 表格报告。
- \`04_mqt_5q_demo.json\`：MQT-Bench 机读结果。
- \`05_ibm_tokyo_topology.png\`：IBM Tokyo 20Q 拓扑图。
- \`06_sabre_heuristic_lab.md\`：basic / lookahead / decay 的可复现对比。
- \`07_bounded_search_negative_sample.md\`：bounded search 负样本。

本地网站演示：

\`\`\`bash
uvicorn src.server.app:app --reload --port 8765
python -m http.server 8766 --directory docs
\`\`\`

打开 \`http://127.0.0.1:8766/index.html\` 后，可选择 example、heuristic 和
inline OpenQASM，并通过 local API 看到 live result 与 static table 是否一致。
EOF

echo
echo "============================================================"
echo "Done. Evidence pack generated at:"
echo "  ${RESULTS_DIR}"
echo "============================================================"

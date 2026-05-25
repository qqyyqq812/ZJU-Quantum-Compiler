# ZJU Quantum Circuit AI Compiler

> **课题四：复杂拓扑结构下的量子电路人工智能编译与动态路由优化**
>
> A Qiskit-compatible quantum circuit routing package that compares a stable
> SABRE baseline with experimental RL/GNN routers on constrained hardware
> topologies.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/qiskit-2.0+-purple.svg)](https://www.ibm.com/quantum/qiskit)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org)

本项目按老师原始任务书收束：开发一个兼容 Qiskit 的开源 Python
量子电路编译器插件，学习真实芯片拓扑约束，动态决定 SWAP 插入策略，
并用电路深度、CNOT/SWAP 数与 Qiskit SABRE 做可复现对比。

当前结论保持诚实：SABRE 是稳定基线；V14/V15 AI checkpoint 已能加载和评测，
但在 2026-05-04 的 P1 主集合上尚未超过 SABRE。项目价值在于完整的
量子路由建模、RL/GNN 工程链路、真实失败诊断和下一代 MCTS 路线。

## 评分对标

原始评分文件 `大作业评价.pdf` 将项目分为五项。仓库按这五项组织证据：

| 评分维度 | 权重 | 本仓库证据 |
| --- | ---: | --- |
| 项目周期管理与早期提交 | 20% | Git commit 历史覆盖 V7-V15，保留训练、评测、修复和纠偏提交 |
| 开源工程规范与文档质量 | 25% | `requirements.txt`、`pyproject.toml`、`qcompiler`、pytest、FastAPI、本 README |
| 物理机制解析与算法架构设计 | 30% | `docs/technical/项目报告.md`、`docs/technical/decisions.md`、`src/compiler/` |
| 开源社区展示度与反馈互动 | 15% | GitHub Pages playground、示例 QASM、notebook、公开 benchmark 表 |
| AI 工具高级调用与协同声明 | 10% | `AI-Collaboration.md` 记录调研、诊断、纠偏和人工决策过程 |

## 老师验收快速路径

推荐用 `requirements.txt` 复现老师验收环境；开发者也可以用
`pip install -e .[dev]` 安装可编辑包。

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

qcompiler info
qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre
qcompiler eval --circuits qft_5,qaoa_5,ghz_5 --topology tokyo
bash run_teacher_eval.sh
```

`run_teacher_eval.sh` 会生成 `results/teacher_demo/`，其中包含环境状态、
SABRE 编译输出、SABRE/AI 小表、MQT-Bench 5Q 报告、IBM Tokyo 拓扑图和报告索引。
AI 列只显示真实的 `OK`、`INCOMPLETE` 或 `N/A`。该脚本中的 5Q 表是现场
smoke demo；项目主结论仍以 12 条 P1 主集合报告为准。

## 公开展示入口

`docs/index.html` 是 GitHub Pages playground，用来服务“开源社区展示度”
和外部用户快速理解项目。它不是项目唯一验收入口。

- GitHub Pages: `https://qqyyqq812.github.io/ZJU-Quantum-Compiler/`
- Local file: [`docs/index.html`](docs/index.html)
- Optional API: `uvicorn src.server.app:app --reload --port 8765`

## 课题四对应实现

老师任务书要求“学习物理拓扑约束，动态决策初始映射和 SWAP 门插入策略，
最小化电路深度和 CNOT 门总数，并与 SABRE 比较”。本项目对应实现如下：

| 要求 | 实现 |
| --- | --- |
| 兼容 Qiskit 的开源 Python 包 | `pyproject.toml` 暴露 `qcompiler`，`src.compiler.pass_manager.AIRouter` 可被 Qiskit 电路调用 |
| 物理拓扑约束 | `src/benchmarks/topologies.py` 提供 IBM Tokyo、linear、grid、heavy-hex 等 coupling map |
| SWAP 动态路由 | `src/compiler/env.py` 将 SWAP 边建模为动作，`CircuitDAG` 判断可执行前沿门 |
| AI 路由器 | `PolicyNetwork` 使用 GraphSAGE + edge-scoring actor，V15 提供 MCTS+GNN POC |
| SABRE 对比 | `scripts/eval_mqt_bench.py` 和 `qcompiler eval` 输出 SABRE/AI SWAP 与完成状态 |
| 可复现展示 | `run_teacher_eval.sh`、`notebooks/05_demo_v14_vs_sabre.ipynb`、GitHub Pages |

## 当前 P1 评测结果

最新公开 P1 评测生成于 2026-05-04，模型为
`models/v14_tokyo20/checkpoint_ep25333.pt`，拓扑为 IBM Tokyo 20Q。主集合覆盖
`qft`、`qaoa`、`ghz`、`vqe` 四类电路在 5、10、20 qubits 上的表现。

| Metric | Result |
| --- | --- |
| SABRE completion | 12/12 |
| AI completion | 4/12 |
| AI beats SABRE | 0/4 completed comparable rows |
| Mean AI/SABRE ratio | 2.500 on completed rows with SABRE > 0 |

完整报告见
[`models/v14_tokyo20/eval_report_mqt.md`](models/v14_tokyo20/eval_report_mqt.md)，
机读数据见
[`models/v14_tokyo20/eval_report_mqt.json`](models/v14_tokyo20/eval_report_mqt.json)。

![V14 training curve](docs/figures/v14_training_curve.png)

## 示例

这些 QASM 示例用于快速验证 SABRE 路径和 API 路径：

| File | Purpose | Suggested command |
| --- | --- | --- |
| [`examples/qft5.qasm`](examples/qft5.qasm) | Dense QFT interactions | `qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre` |
| [`examples/ghz5.qasm`](examples/ghz5.qasm) | Small entanglement chain | `qcompiler compile examples/ghz5.qasm --topology tokyo --backend sabre` |
| [`examples/qaoa5.qasm`](examples/qaoa5.qasm) | One QAOA-style layer | `qcompiler eval --circuits qaoa_5 --topology tokyo` |

运行 AI 状态对比：

```bash
qcompiler eval \
    --model models/v14_tokyo20/checkpoint_ep25333.pt \
    --circuits qft_5,qaoa_5,ghz_5 \
    --topology tokyo
```

## 本地 API

```bash
uvicorn src.server.app:app --reload --port 8765
```

| Endpoint | Description |
| --- | --- |
| `GET /api/status` | Version, topology aliases, model path, and AI load status |
| `GET /api/examples` | Public QASM example list |
| `POST /api/compile` | Compile an example or inline OpenQASM with `backend=sabre\|ai` |
| `GET /api/benchmarks` | Return the checked-in V14 P1 summary without rerunning it |

## 复现 P1 报告

```bash
python scripts/eval_mqt_bench.py \
    --ai-model models/v14_tokyo20/checkpoint_ep25333.pt \
    --n-qubits 5,10,20 \
    --benchmarks qft,qaoa,ghz,vqe \
    --output models/v14_tokyo20/eval_report_mqt.md
```

## 算法演进

| Version | Algorithm | Main change | Current status |
| --- | --- | --- | --- |
| V9 | PPO baseline | IBM Tokyo 20Q with hard masks | Historical convergence run |
| V10 | PPO + hard mask | Closed the soft-constraint loophole | Historical |
| V11 | DQN | Compared against PPO | Historical |
| V13 | PPO + GNN 9D | SABRE-relative reward and pure PyTorch GraphSAGE | Stage 1 unstable |
| V14 | PPO + GNN fixes | SABRE cache, staged masks, true pass-manager integration | P1 did not pass |
| V15 | AlphaZero-style MCTS + GNN | Self-play and MCTS research path | Short POC only |

详见 [`docs/technical/decisions.md`](docs/technical/decisions.md)。

## 项目结构

```text
ZJU-Quantum-Compiler/
├── docs/                  # GitHub Pages site and technical notes
├── examples/              # Public QASM examples
├── src/
│   ├── benchmarks/        # Circuits, topologies, and MQT-Bench helpers
│   ├── compiler/          # Env, policy, AIRouter, V15 research code
│   ├── server/            # Optional local FastAPI playground backend
│   └── cli.py             # qcompiler command line interface
├── scripts/               # Evaluation and plotting scripts
├── models/v14_tokyo20/    # Public report JSON/Markdown; weights are ignored
├── tests/                 # Pytest smoke and regression tests
└── pyproject.toml
```

## 开发检查

```bash
pytest -q
python -m py_compile src/cli.py src/server/app.py scripts/eval_mqt_bench.py
git diff --check
```

## References

- Li, G. et al. "Tackling the Qubit Mapping Problem for NISQ-Era Quantum
  Devices." ASPLOS 2019.
- Hayes, J. et al. "LightSABRE: A Lightweight and Enhanced SABRE Algorithm."
  2024.
- Sinha, A. et al. "Qubit routing using graph neural network aided Monte Carlo
  tree search." AAAI 2022.
- Park, S. et al. "AlphaRouter: Quantum Circuit Routing with Reinforcement
  Learning and MCTS." 2024.

## License

MIT. See [`LICENSE`](LICENSE).

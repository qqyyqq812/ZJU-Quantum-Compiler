# ZJU Quantum Circuit AI Compiler

> **课题四：复杂拓扑结构下的量子电路人工智能编译与动态路由优化**
>
> A Qiskit-compatible quantum circuit routing package and practical heuristic
> lab for comparing SabreSwap / LightSABRE routing choices on constrained
> hardware topologies.

`Python 3.10+` · `Qiskit 2.0+` · `PyTorch 2.0+` · `MIT License`

本项目按老师原始任务书收束：开发一个兼容 Qiskit 的开源 Python
量子电路编译器插件，学习真实芯片拓扑约束，动态决定 SWAP 插入策略，
并用电路深度、CNOT/SWAP 数与 Qiskit SABRE 做可复现对比。

当前结论保持诚实：SABRE / LightSABRE 是稳定实用基线；V14/V15 AI
checkpoint 已能加载和评测，但在 2026-05-04 的 P1 主集合上尚未超过 SABRE。
2026-05-29 后，公开网站和本地 API 的实用主线已转为 SabreSwap heuristic
lab：默认使用 `lookahead + seed=42 + trials=1`，同时保留 AI 路由器历史作为
工作量和失败复盘证据。

## 评分对标

原始评分文件 `大作业评价.pdf` 将项目分为五项。仓库按这五项组织证据：

| 评分维度 | 权重 | 本仓库证据 |
| --- | ---: | --- |
| 项目周期管理与早期提交 | 20% | Git commit 历史覆盖 V7-V15，保留训练、评测、修复和纠偏提交 |
| 开源工程规范与文档质量 | 25% | `requirements.txt`、`pyproject.toml`、`qcompiler`、pytest、FastAPI、本 README |
| 物理机制解析与算法架构设计 | 30% | `docs/technical/项目报告.md`、`docs/technical/decisions.md`、`src/compiler/` |
| 开源社区展示度与反馈互动 | 15% | GitHub Pages heuristic lab、示例 QASM、notebook、公开 benchmark 表 |
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
qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre --heuristic lookahead
qcompiler eval --circuits qft_5,qaoa_5,ghz_5 --topology tokyo
bash run_teacher_eval.sh
```

`run_teacher_eval.sh` 会生成 `results/teacher_demo/`，其中包含环境状态、
SABRE 编译输出、SABRE/AI 小表、MQT-Bench 5Q 报告、IBM Tokyo 拓扑图、
SabreSwap heuristic lab 表、bounded search 负样本和报告索引。AI 列只显示
真实的 `OK`、`INCOMPLETE` 或 `N/A`。该脚本中的 5Q 表是现场 smoke demo；
项目主结论仍以 12 条 P1 主集合报告为准。

## 公开工具入口

`docs/index.html` 是 GitHub Pages heuristic lab，用来服务量子信息大作业的
实用网站目标和算法大作业的展示证据链。它不是项目唯一验收入口；CLI、
FastAPI、本地脚本和报告共同构成可复现证据。

- GitHub Pages: `https://qqyyqq812.github.io/ZJU-Quantum-Compiler/`
- Local file: [`docs/index.html`](docs/index.html)
- Optional API: `uvicorn src.server.app:app --reload --port 8765`

当前网站支持：

- 选择 checked-in QASM examples。
- 粘贴小型 inline OpenQASM 2。
- 比较 `basic`、`lookahead`、`decay` 三种 SabreSwap heuristic。
- 调用本地 API，显示 `status`、`SWAP`、`depth` 和 static/live match。
- 明确说明 AI-router 历史只作为工作量和失败分析证据，不作为性能胜利宣传。

## 现场演示 checklist

课堂或答辩现场建议按下面顺序演示。目标是证明网站是可用工具，而不是静态截图。

1. 启动本地 API：

   ```bash
   uvicorn src.server.app:app --reload --port 8765
   ```

2. 启动静态网站服务：

   ```bash
   python -m http.server 8766 --directory docs
   ```

3. 打开 `http://127.0.0.1:8766/index.html`。
4. 选择 `QFT 10` 和 `lookahead`，点击 **Run local API**。
5. 核对结果卡片：
   - `status` 应为 `OK`。
   - `SWAP` 应为 `29`。
   - `match` 应为 `yes`。
6. 切换 `basic` 或 `decay`，展示同一电路下不同 heuristic 的 SWAP 差异。
7. 粘贴或填入 mini GHZ OpenQASM，说明网站支持自定义输入。

如果现场 API 无法启动，仍可直接打开 `docs/index.html` 或 GitHub Pages。此时页面
会进入 static mode，仍能展示 checked-in examples、静态 heuristic 表和项目口径；
但不能声称完成了 live API 复现。

## 课题四对应实现

老师任务书要求“学习物理拓扑约束，动态决策初始映射和 SWAP 门插入策略，
最小化电路深度和 CNOT 门总数，并与 SABRE 比较”。本项目对应实现如下：

| 要求 | 实现 |
| --- | --- |
| 兼容 Qiskit 的开源 Python 包 | `pyproject.toml` 暴露 `qcompiler`，`src.compiler.pass_manager.AIRouter` 可被 Qiskit 电路调用 |
| 物理拓扑约束 | `src/benchmarks/topologies.py` 提供 IBM Tokyo、linear、grid、heavy-hex 等 coupling map |
| SWAP 动态路由 | `src/compiler/env.py` 将 SWAP 边建模为动作，`CircuitDAG` 判断可执行前沿门 |
| AI 路由器 | `PolicyNetwork` 使用 GraphSAGE + edge-scoring actor，V15 提供 MCTS+GNN POC |
| SABRE / LightSABRE 对比 | `scripts/eval_mqt_bench.py`、`scripts/experiment_sabre_trials.py` 和 `qcompiler eval` 输出 SABRE/AI/heuristic 证据 |
| 可复现展示 | `run_teacher_eval.sh`、`notebooks/05_demo_v14_vs_sabre.ipynb`、GitHub Pages heuristic lab |

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

这些 QASM 示例用于快速验证 SABRE 路径、API 路径和网站静态表口径：

| File | Purpose | Suggested command |
| --- | --- | --- |
| [`examples/qft5.qasm`](examples/qft5.qasm) | Dense QFT interactions | `qcompiler compile examples/qft5.qasm --topology tokyo --backend sabre --heuristic lookahead` |
| [`examples/ghz5.qasm`](examples/ghz5.qasm) | Small entanglement chain | `qcompiler compile examples/ghz5.qasm --topology tokyo --backend sabre --heuristic lookahead` |
| [`examples/qaoa5.qasm`](examples/qaoa5.qasm) | One QAOA-style layer | `qcompiler eval --circuits qaoa_5 --topology tokyo` |
| [`examples/qft10.qasm`](examples/qft10.qasm) | Larger dense QFT routing pressure | `qcompiler compile examples/qft10.qasm --topology tokyo --backend sabre --heuristic lookahead` |
| [`examples/qaoa10.qasm`](examples/qaoa10.qasm) | 10-qubit QAOA-style layer | `qcompiler compile examples/qaoa10.qasm --topology tokyo --backend sabre --heuristic lookahead` |
| [`examples/ghz10.qasm`](examples/ghz10.qasm) | 10-qubit entanglement chain | `qcompiler compile examples/ghz10.qasm --topology tokyo --backend sabre --heuristic lookahead` |
| [`examples/vqe10.qasm`](examples/vqe10.qasm) | 10-qubit VQE-like ansatz | `qcompiler compile examples/vqe10.qasm --topology tokyo --backend sabre --heuristic lookahead` |

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
| `POST /api/compile` | Compile an example or inline OpenQASM with `backend=sabre\|ai`; SABRE accepts `heuristic=basic\|lookahead\|decay` |
| `GET /api/benchmarks` | Return the checked-in V14 P1 summary without rerunning it |

Minimal live heuristic call:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"qft10","backend":"sabre","topology":"tokyo","heuristic":"lookahead"}'
```

Inline OpenQASM 2 input is limited to 8000 characters to keep the public API
path suitable for classroom demos and local experimentation.

The CLI `compile` command uses the same SABRE heuristic choices as the API:
`--heuristic basic|lookahead|decay`. The default is `lookahead`, with `seed=42`
and `trials=1`, to match the website static/live table.

## 算法筛选实验

当前算法大作业不是按教学课件组织，而是展示问题、候选算法、实现工作量、
实验结果和失败复盘。已保留两类本地实验脚本：

| Script | Purpose | Current use |
| --- | --- | --- |
| `scripts/experiment_sabre_trials.py` | Compare `basic`、`lookahead`、`decay` across trial counts | 解释为什么网站默认使用快速可复现的 `lookahead + trials=1` |
| `scripts/experiment_bounded_search.py` | Test a deterministic bounded-search prototype | 负样本：`qft10` 上明显差于 `lookahead`，不进入网站默认算法 |

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
git diff --check
python -m py_compile \
    src/cli.py \
    src/server/app.py \
    scripts/eval_mqt_bench.py \
    scripts/experiment_sabre_trials.py \
    scripts/experiment_bounded_search.py
python -m pytest -s -q \
    tests/test_experiment_scripts.py \
    tests/test_public_site.py \
    tests/test_public_api.py \
    tests/test_cli.py \
    tests/test_v15_smoke.py
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

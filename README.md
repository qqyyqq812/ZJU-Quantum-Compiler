# ZJU Quantum Compiler

语言 / Language: **中文** | [English](#english-version)

本仓库是量子信息基础大作业的公开项目入口。项目实现了一个面向受限
硬件拓扑的量子线路路由编译器：输入 OpenQASM 2 线路，选择硬件耦合
拓扑，完成逻辑量子位到物理量子位的映射，插入必要的 SWAP 门，并输出
满足硬件连边约束的路由后线路。默认路由方法为 NPQR，SABRE basic 作为
固定对照基线。

## 快速入口

第一次查看建议按下面顺序进行：

1. 打开在线页面：
   <https://qqyyqq812.github.io/ZJU-Quantum-Compiler/>
2. 选择 `ghz5`、`qft5` 或 `qaoa5`。
3. 点击 **Run**，查看 NPQR 与 SABRE basic 的 SWAP 数、深度和耗时对比。
4. 使用轨迹回放查看 QASM 输入、硬件拓扑、SWAP 插入和路由后线路。
5. 阅读报告 PDF：`docs/report_latex/main.pdf`。

5Q 和 10Q 示例适合快速体验。30Q 和 50Q 示例属于扩展规模，需要部署
后端服务后进行真实编译。GitHub Pages 是给人看的网页入口；REST API 是
网页和脚本调用的真实编译后端；MCP 面向工具客户端和自动化流程，不是
普通浏览器体验的必需入口。

## 三类使用路径

| 路径 | 第一步 | 主要入口 |
| --- | --- | --- |
| 普通体验 | 打开 GitHub Pages，运行 5Q/10Q 示例 | QASM 输入、拓扑图、轨迹回放、指标对比 |
| 本地部署 | 克隆仓库，安装依赖，启动 REST 服务 | README、REST API、网页 |
| 工具调用 | 启动 MCP 或命令行工具 | MCP 工具、`qcompiler` |

查看项目时，重点关注页面是否能打开，QASM 输入是否清楚，拓扑和 SWAP
回放是否解释了编译过程，NPQR/SABRE basic 对比是否明确，以及 README
命令是否能复现同一套公开入口。

## 仓库内容

- `src/`：NPQR 路由运行时、编译工具、REST API 和 MCP 服务。
- `examples/`：5、10、30、50 比特 OpenQASM 示例。
- `models/default/npqr-default.pt`：默认 NPQR 推理模型。
- `docs/index.html`：GitHub Pages 在线编译演示页面。
- `docs/项目说明.md`：中文技术说明。
- `docs/playground-user-guide.md`：网页体验指南。
- `docs/ai-collaboration.md`：AI 协作说明。
- `docs/report_latex/main.pdf`：量子信息基础大作业报告 PDF。
- `scripts/`：复现、检查和打包脚本。
- `tests/`：公开 API、网页、文档和发布契约测试。

## 安装

建议使用 Python 3.10 或更新版本。

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

检查命令行和默认模型：

```bash
qcompiler info
```

## REST API

启动 FastAPI 服务：

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8765
```

主要接口如下：

| 接口 | 用途 |
| --- | --- |
| `GET /api/status` | 返回后端、模型和默认编译状态。 |
| `GET /api/examples` | 列出内置 OpenQASM 示例和默认拓扑。 |
| `POST /api/validate` | 校验 OpenQASM 与所选拓扑是否匹配。 |
| `POST /api/compile` | 编译内置示例或用户输入线路。 |
| `POST /api/compile/jobs` | 创建异步编译任务并记录阶段耗时。 |
| `GET /api/compile/jobs/{job_id}` | 查询异步编译任务。 |
| `GET /api/topology/{name}` | 返回拓扑信息和连边。 |
| `GET /api/benchmarks` | 返回公开实验摘要。 |
| `GET /api/npqr/evidence` | 返回机器可读的 NPQR 证据。 |

编译一个 5Q 示例：

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

编译一个 50Q 示例：

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"line_ghz50","backend":"sabre","topology":"grid_5x10"}'
```

返回结果包含路由后 QASM、SWAP 数、线路深度、耗时、轨迹事件、
初始/最终映射，以及 SABRE basic 对照指标。

## MCP 服务

MCP 服务面向工具客户端和自动化流程。普通网页体验不需要启动 MCP。

```bash
qcompiler-mcp-http
```

默认地址：

```text
http://127.0.0.1:8000/mcp
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

主要 MCP 工具如下：

| 工具 | 用途 |
| --- | --- |
| `qcompiler_status` | 返回编译器、模型和服务状态。 |
| `list_examples` | 列出内置 QASM 示例。 |
| `compile_npqr` | 使用 NPQR 编译内置示例。 |
| `compile_sabre` | 使用 SABRE 编译内置示例。 |
| `compile_qasm` | 编译用户提供的 OpenQASM 2 文本。 |
| `get_benchmarks` | 返回公开实验摘要。 |
| `get_npqr_boundary` | 返回评价范围说明。 |
| `get_algorithm_evidence` | 返回算法组件证据。 |

## 命令行

查看安装状态：

```bash
qcompiler info
```

编译内置示例：

```bash
qcompiler compile --example qft5 --backend npqr
```

编译 QASM 文件：

```bash
qcompiler compile examples/qft5.qasm --backend sabre --topology tokyo
```

生成可复现实验矩阵：

```bash
qcompiler matrix --quick
```

## 示例与拓扑

小规模和中等规模示例默认使用 IBM Tokyo 20Q。扩展规模示例使用与线路
规模匹配的规则网格拓扑。

| 示例 | 比特数 | 默认拓扑 | 文件 |
| --- | ---: | --- | --- |
| QFT 5 | 5 | `tokyo` | `examples/qft5.qasm` |
| GHZ 5 | 5 | `tokyo` | `examples/ghz5.qasm` |
| QAOA 5 | 5 | `tokyo` | `examples/qaoa5.qasm` |
| QFT 10 | 10 | `tokyo` | `examples/qft10.qasm` |
| QAOA 10 | 10 | `tokyo` | `examples/qaoa10.qasm` |
| GHZ 10 | 10 | `tokyo` | `examples/ghz10.qasm` |
| VQE-like 10 | 10 | `tokyo` | `examples/vqe10.qasm` |
| LineGHZ30 | 30 | `grid_5x6` | `examples/line_ghz30.qasm` |
| Random30-d4 | 30 | `grid_5x6` | `examples/random30_d4.qasm` |
| LineGHZ50 | 50 | `grid_5x10` | `examples/line_ghz50.qasm` |
| RingSparse50 | 50 | `grid_5x10` | `examples/ring_sparse50.qasm` |

## 方法概述

量子线路路由可以看作图约束下的近似优化问题。物理量子位是图顶点，
硬件耦合关系是图边。双比特门只有在两个逻辑量子位当前映射到相邻
物理顶点时才能直接执行，否则编译器需要沿合法硬件边插入 SWAP。

NPQR 流程如下：

1. 解析 OpenQASM 线路和目标硬件拓扑。
2. 构建门依赖关系并识别前沿门。
3. 生成逻辑量子位到物理量子位的初始映射候选。
4. 使用神经模型对合法 SWAP 动作评分。
5. 使用有界束搜索保留多条候选路线。
6. 对困难局部结构触发更强搜索和剪枝。
7. 在接近完成但停滞的状态中使用局部修复。
8. 复放最终轨迹并输出通过拓扑验证的 QASM。

## 实验结果

公开证据使用 SABRE basic 作为固定质量基线。报告中的主实验覆盖
代表性 10/20Q 示例，扩展实验覆盖选定 30/50Q 示例。网页优先推荐
5Q 和 10Q 示例，因为它们运行快、轨迹短、便于观察。

| 规模 | 用例数 | 作用 |
| --- | ---: | --- |
| 5/10Q 网页示例 | 7 | 快速体验和页面展示。 |
| 10/20Q 代表用例 | 10 | 报告主实验。 |
| 30/50Q 扩展用例 | 4 | 需要后端支持的扩展规模实验。 |

## 部署

构建并运行 REST API 容器：

```bash
docker build -f Dockerfile.api -t zju-quantum-compiler-api .
docker run --rm -p 8080:8080 zju-quantum-compiler-api
```

构建并运行 MCP 容器：

```bash
docker build -f Dockerfile.mcp -t zju-quantum-compiler-mcp .
docker run --rm -p 8081:8081 zju-quantum-compiler-mcp
```

Render 部署配置见 `render.yaml` 和 `render-mcp.yaml`。

## 验证

发布前可运行以下检查：

```bash
python -m pytest -q \
  tests/test_public_api.py \
  tests/test_public_site.py \
  tests/test_public_deploy_config.py \
  tests/test_readme_contract.py \
  tests/test_submission_readiness.py \
  tests/test_algorithm_matrix.py \
  tests/test_demo_scripts.py \
  tests/test_playground_user_guide.py
python scripts/check_submission_readiness.py
git diff --check
```

<details id="english-version">
<summary>English version</summary>

# ZJU Quantum Compiler

ZJU Quantum Compiler is the public repository for the Quantum Information
Foundations final work. It implements a quantum circuit routing compiler for
restricted hardware topologies. The compiler accepts OpenQASM 2 circuits,
maps logical qubits to physical qubits, inserts required SWAP gates, and
returns a routed circuit that satisfies the target coupling graph. The default
compiler path is NPQR, and SABRE basic is the fixed Qiskit baseline.

## Quick entry

Use this path for the first pass:

1. Open the GitHub Pages console:
   <https://qqyyqq812.github.io/ZJU-Quantum-Compiler/>
2. Select `ghz5`, `qft5`, or `qaoa5`.
3. Click **Run** and compare NPQR with SABRE basic.
4. Step through the route trace to inspect QASM input, topology, inserted
   SWAP operations, depth, and routed QASM output.
5. Read `docs/report_latex/main.pdf` for the report and experiment tables.

GitHub Pages is the human-facing page. REST API performs real compiler calls
for the page and scripts. MCP is an advanced interface for tool clients and
automation.

## Paths

| Path | First action | Main surface |
| --- | --- | --- |
| Browser experience | Open GitHub Pages and run 5Q/10Q examples | QASM editor, topology, trace replay, metrics |
| Local deployment | Clone the repository, install dependencies, run REST | README, REST API, GitHub Pages |
| Tool workflow | Start MCP or use the command line | MCP tools and `qcompiler` |

## Install

Use Python 3.10 or newer.

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
qcompiler info
```

## Run the REST API

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8765
```

Compile a small checked-in example:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

## Run MCP

```bash
qcompiler-mcp-http
curl -s http://127.0.0.1:8000/health
```

The MCP endpoint is `http://127.0.0.1:8000/mcp`.

## Verify

```bash
python scripts/check_submission_readiness.py
git diff --check
```

</details>

# ZJU Quantum Compiler

这是一个面向受限量子硬件的神经辅助量子路由编译器。输入
OpenQASM 2 电路后，系统会把逻辑量子比特映射到真实芯片拓扑上，
插入必要的 SWAP 门，并输出满足硬件连接约束的路由后电路。

默认编译流程是 NPQR。NPQR 使用神经网络进行动作评分，并结合初始映射
选择、束搜索、动作剪枝和局部修复来完成路由。SABRE 作为固定对比基线，
用于展示传统启发式算法的效果，不参与 NPQR 的路线生成。

## 项目内容

仓库中的主要内容如下：

- `src/`：量子路由编译器、NPQR 运行时、REST API 和 MCP 服务。
- `examples/`：可直接运行的 OpenQASM 示例电路。
- `models/default/npqr-default.pt`：默认 NPQR 推理模型。
- `docs/`：网页、项目说明、PPT 和报告分工材料。
- `scripts/`：可复现实验矩阵、提交检查和材料打包脚本。
- `tests/`：公开接口、网页、部署、文档和检查脚本的测试。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

检查环境和默认模型状态：

```bash
qcompiler info
```

## 启动 REST API

REST API 适合网页、脚本和普通后端调用。

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8765
```

常用接口如下：

| 接口 | 作用 |
| --- | --- |
| `GET /api/status` | 查看后端状态、默认编译器、模型路径和模型加载状态。 |
| `GET /api/examples` | 获取仓库内置的 OpenQASM 示例。 |
| `POST /api/compile` | 使用 NPQR 或 SABRE 编译示例电路或自定义 QASM。 |
| `GET /api/benchmarks` | 查看算法边界和固定示例摘要。 |
| `GET /api/npqr/evidence` | 获取机器可读的 NPQR 算法说明。 |

使用默认 NPQR 编译内置示例：

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

使用默认 NPQR 编译自定义 OpenQASM：

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"qasm":"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"}'
```

返回结果包含路由后的 QASM、SWAP 数、电路深度、运行时间、路线轨迹和
SABRE 对比结果。`components` 字段中的 `sabre_fallback` 为 `false`，
表示 NPQR 结果不是由 SABRE 代替完成的。

## 启动 MCP 服务

MCP 服务适合工具客户端和自动化调用。

```bash
qcompiler-mcp-http
```

默认 MCP 地址为：

```text
http://127.0.0.1:8000/mcp
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

主要 MCP 工具如下：

| 工具 | 作用 |
| --- | --- |
| `qcompiler_status` | 查看默认编译器、模型和算法边界。 |
| `list_examples` | 列出内置 QASM 示例。 |
| `compile_npqr` | 使用 NPQR 编译内置示例。 |
| `compile_sabre` | 使用 SABRE 编译内置示例。 |
| `compile_qasm` | 编译用户提供的 OpenQASM 2 文本。 |
| `get_benchmarks` | 返回公开的基准和声明边界。 |
| `get_npqr_boundary` | 说明 NPQR 的能力范围和不声明内容。 |
| `get_algorithm_evidence` | 返回面向课程报告的算法组成说明。 |

## 命令行用法

查看项目信息：

```bash
qcompiler info
```

编译示例电路：

```bash
qcompiler compile --example qft5 --backend npqr
```

生成快速对比矩阵：

```bash
qcompiler matrix --quick
```

## 算法说明

量子芯片可以看成一个无向图，物理量子比特是顶点，硬件连接是边。
输入电路中的双比特门只有在对应物理量子比特相邻时才能直接执行。
如果不相邻，编译器需要插入 SWAP 门来移动逻辑量子态。

NPQR 的流程如下：

1. 解析 OpenQASM 电路和 IBM Tokyo 20Q 拓扑。
2. 构建门依赖关系，识别当前可执行的前沿门。
3. 生成逻辑量子比特到物理量子比特的初始映射候选。
4. 使用神经网络对合法 SWAP 动作进行评分。
5. 使用有界束搜索保留多条候选路线，降低一步贪心失误的影响。
6. 在较难的交互模式上触发更强的前沿搜索。
7. 通过动作剪枝控制候选规模和运行时间。
8. 在路线接近完成但局部卡住时进行有界后缀修复。
9. 复放最终轨迹，确认每一步都满足芯片拓扑约束，再输出 QASM。

SABRE 是对比基线。它是 Qiskit 中常用的量子路由启发式方法，主要根据
前沿门和后续门的距离估计选择 SWAP。项目中保留 SABRE 是为了进行稳定、
可复现的对比。

## 课程算法概念对应

项目可以用下列算法设计概念解释：

| 课程概念 | 在项目中的体现 |
| --- | --- |
| 图问题 | 芯片拓扑是图，量子比特是顶点，硬件连接是边。 |
| 变治法 | 把电路路由转化为图约束下的映射和路径规划问题。 |
| 贪婪思想 | 前沿门距离的下降可作为局部 SWAP 质量判断。 |
| 减治法 | 每执行一个合法门，剩余待路由任务就减少。 |
| 时空权衡 | 距离矩阵、候选映射和候选路线会占用空间，但减少重复计算。 |
| 迭代改进 | 路线通过 SWAP、映射更新和局部修复逐步改进。 |
| 搜索剪枝 | 束宽、动作剪枝和触发规则限制搜索树规模。 |
| 近似求解 | 系统追求有限时间内的高质量可行解，不保证全局最优。 |
| 神经网络 | 模型在搜索过程中提供动作偏好。 |

## 示例电路

仓库内置示例覆盖不同路由压力：

| 示例 | 说明 |
| --- | --- |
| `examples/ghz5.qasm` | 小规模纠缠链。 |
| `examples/qft5.qasm` | 小规模密集交互电路。 |
| `examples/qaoa5.qasm` | 小规模 QAOA 风格电路。 |
| `examples/qft10.qasm` | 路由压力更高的密集交互电路。 |
| `examples/qaoa10.qasm` | 中等规模优化风格电路。 |
| `examples/ghz10.qasm` | 更长的纠缠链。 |
| `examples/vqe10.qasm` | VQE 风格 ansatz。 |

运行快速矩阵：

```bash
python scripts/experiment_algorithm_matrix.py --quick
```

生成 JSON 结果，便于整理报告表格：

```bash
python scripts/experiment_algorithm_matrix.py --quick --json
```

## 网页和报告材料

`docs/index.html` 是唯一网页入口。GitHub Pages 打开后默认使用内置示例结果，
可以直接展示 NPQR 与固定 SABRE 基线的对比、Tokyo 映射和路线回放。维护者
可以用 HTTPS REST API 覆盖为实时编译。

`docs/playground-user-guide.md` 说明网页中的 **Run**、**Step**、**Reset**、
示例电路、自定义 QASM、生成电路、Tokyo 映射、路线轨迹和 `compiled_qasm`
结果面板。

`docs/项目说明.md` 提供更完整的中文项目说明，覆盖背景、算法、接口、
目录结构、运行方式、测试和结果边界。

`docs/plans/组员分工.md` 是组员报告分工说明，内容围绕算法流程、课程概念、
复杂度分析、对比基线、PPT 结构和口播材料展开。

PPT 材料位于 `docs/slides/`。

## 结果摘要

本课程项目把 SABRE basic 作为主要质量基线。最终本地评测显示，NPQR 在
代表性 10/20 比特电路上全部完成路由，并且 SWAP 数均低于 SABRE basic。
所有完成结果都不依赖 SABRE 回退路径，并通过轨迹复放验证。

| 规模 | 电路数 | NPQR 完成 | 优于 SABRE basic | 说明 |
| --- | ---: | ---: | ---: | --- |
| 10/20 比特代表电路 | 10 | 10 | 10 | 主性能结论。 |
| 30/50 比特扩展测试 | 4 | 4 | 4 | 展示扩展潜力，但不是全面胜出声明。 |
| 80/100 比特边界测试 | 4 | 0 | 0 | 用于界定上限，不作为完成能力声明。 |

30/50 比特扩展测试中，NPQR 在 LineGHZ30、Random30-d4、LineGHZ50 和
RingSparse50 上均完成并优于 SABRE basic。
80 比特 LineGHZ 在 240 秒 CPU 有界运行内未完成，100 比特没有作为最终能力
声明。因此，项目可以宣称 NPQR 在代表性 10/20 比特任务上稳定优于 SABRE basic，
并已证明部分 30/50 比特电路可完成且优于 basic；当前最大已证明完成规模为
50 比特，不能宣称所有大规模电路都优于 SABRE。

## 部署

REST API 镜像：

```bash
docker build -f Dockerfile.api -t zju-quantum-compiler-api .
docker run --rm -p 8080:8080 zju-quantum-compiler-api
```

MCP 镜像：

```bash
docker build -f Dockerfile.mcp -t zju-quantum-compiler-mcp .
docker run --rm -p 8081:8081 zju-quantum-compiler-mcp
```

Render 配置文件为 `render.yaml` 和 `render-mcp.yaml`。

## 检查命令

发布或演示前可以运行以下检查：

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

生成本地材料包：

```bash
python scripts/package_submission.py
```

生成文件位于 `results/submission_package/`。

## 结果边界

NPQR 是一个可运行的神经辅助量子路由流程。它的核心是神经网络推理，但完整
路由结果依赖映射选择、搜索、剪枝、修复和轨迹校验共同完成。

项目不声明 NPQR 在所有电路上都优于 SABRE，也不声明默认模型是最先进的量子
路由模型。SABRE 在这里是强基线和对比对象，不是 NPQR 结果的隐藏完成路径。
SABRE lookahead 不是本课程项目的主比较目标。

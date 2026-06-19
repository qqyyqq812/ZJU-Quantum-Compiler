# 网页体验指南

本指南说明量子信息基础大作业网页如何使用。页面用于查看 OpenQASM
输入、硬件拓扑、SWAP 轨迹回放、路由后 QASM，以及 NPQR 与 SABRE
basic 的指标对比。

## 三分钟体验

第一次打开页面时，建议先运行小规模示例。

1. 打开 GitHub Pages 页面或本地网页。
2. 选择 `ghz5`、`qft5` 或 `qaoa5`。
3. 点击 **Run**。
4. 查看 NPQR 和 SABRE basic 两列的状态、SWAP 数、深度和 `elapsed_ms`。
5. 点击 **Step**，按步骤查看 `route_trace`。
6. 在 **NPQR QASM** 和 **SABRE QASM** 之间切换，查看 `compiled_qasm`。
7. 对照拓扑图，观察每个双比特门或 SWAP 使用的物理连边。

5Q 和 10Q 示例适合首次体验，轨迹短，拓扑关系清楚。30Q 和 50Q 示例是
扩展规模；`LineGHZ30` 与 `Random30-d4` 使用 `grid_5x6`，`LineGHZ50`
与 `RingSparse50` 使用 `grid_5x10`。这些示例需要 REST API 后端支持。

## 页面与后端

网页入口和编译后端是两层。

| 入口 | 面向对象 | 作用 |
| --- | --- | --- |
| GitHub Pages | 浏览器用户 | 打开网页界面，查看输入、拓扑、回放和结果。 |
| REST API | 网页、脚本、部署环境 | 执行真实 NPQR 和 SABRE basic 编译。 |
| MCP | 工具客户端和自动化流程 | 面向进阶工具调用，不是普通网页体验的必需入口。 |

服务面板里的 **运行位置** 决定 REST API 地址：

- **公共服务器** 使用当前部署的在线后端。
- **本地后端** 使用 `http://127.0.0.1:8765`。
- **自定义地址** 用于测试自己的 REST API 部署。

网页可以从 GitHub Pages 打开，也可以在本地打开。公共服务器不可用时，
克隆仓库并启动本地 REST API，然后用本地网页地址进入：

```text
http://127.0.0.1:5500/?mode=local
```

也可以用 GitHub Pages 地址加 `?mode=local` 进入本地后端模式。如果在线
页面调用本地 HTTP 后端受浏览器限制，使用本地网页地址。

## 输入区域

左侧面板控制电路输入。

- **Examples** 载入仓库内置 OpenQASM 示例。5Q 和 10Q 示例使用
  IBM Tokyo 20Q，30Q 和 50Q 示例使用网格拓扑。
- **Custom QASM** 用于粘贴 OpenQASM 2 文本，第一行需要是
  `OPENQASM 2.0;`。
- **Generate** 生成一个小型 OpenQASM 电路，生成后点击 **Run** 编译。

## 结果区域

结果列展示主要指标。

- **Status** 表示路由是否完成。
- **SWAP** 表示插入的 SWAP 门数量。
- **Depth** 表示路由后线路深度。
- **elapsed_ms** 表示本次请求的后端计算耗时。
- **Delta values** 表示 NPQR 与 SABRE basic 的差值。

`compiled_qasm` 面板展示路由后的输出线路。使用 **NPQR QASM** 和
**SABRE QASM** 可以切换两种输出。

## 拓扑与轨迹

拓扑图展示物理量子位和硬件连边。映射后的两个物理量子位相邻时，双比特
门可以直接执行；不相邻时，编译器需要沿合法连边插入 SWAP。

后端返回的 `route_trace` 记录每一步路由事件。

- `gate` 表示当前映射下可以执行的门。
- `swap` 表示为了后续门插入的 SWAP。
- `logical_qubits` 和 `physical_qubits` 标识逻辑量子位与物理量子位。
- `mapping_before` 和 `mapping_after` 展示 SWAP 前后的映射变化。

轨迹标签页可以分别查看 NPQR 和 SABRE basic 的路由过程。

## 后端方法

NPQR 是项目默认路由方法。它解析 OpenQASM，建立门依赖，生成初始映射
候选，筛选合法 SWAP 动作，进行动作评分和有界搜索，并在返回结果前复放
轨迹。

SABRE basic 是固定 Qiskit baseline。两种方法使用同一输入线路、同一拓扑
和同一组指标字段。

## 常见问题

- 页面能打开但编译失败时，先检查服务面板中的 REST API 地址。
- 本地运行时，先启动 `uvicorn` 后端，再打开本地网页。
- GitHub Pages 页面调用本地 HTTP 后端受限时，改用本地网页地址。
- 自定义 QASM 立即失败时，检查第一行是否为 `OPENQASM 2.0;`。
- 30Q 或 50Q 示例耗时较长时，先使用 5Q 或 10Q 示例确认流程。

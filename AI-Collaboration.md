# AI 协同日志

本文件对应老师评分表中的“AI 工具高级调用与协同声明”一项。项目使用
Claude Code、Codex 和并行 subagent 作为协同研究员，主要承担资料检索、
代码审查、实验脚本修复、训练异常诊断和文档证据整理。项目方向、取舍和
最终口径由学生人工确认。

核心原则是：AI 可以加速工程执行，但不能替代真实评测；当模型没有超过
SABRE 时，文档必须如实记录失败和原因。

## 1. 协同方式概览

| 协同方式 | 项目中的用途 | 人工决策点 |
| --- | --- | --- |
| 连续追问 | 追问训练为什么发散、评测为什么全是 N/A、网页口径是否偏离原题 | 选择回到老师原始评分主线 |
| 逻辑重构 | 将项目从 V7/V9 展示叙事重排为“物理机制-算法-评测-诊断” | 将 public playground 升级为服务算法主线的 heuristic lab |
| Bug 定位 | AIRouter checkpoint 加载、pass manager 假集成、V15 self-play 串行 | 确认先修 P1 评测闭环，再谈 V15 长训 |
| 文献和路线调研 | 对比 SABRE、LightSABRE、MCTS+GNN、AlphaRouter 路线 | 从 PPO 长训转向 bounded MCTS/GNN POC |
| 证据整理 | README、项目报告、评测报告、老师验收脚本 | 拒绝“AI 已打败 SABRE”的无证据包装 |
| 工具化收口 | 将网页、API、QASM 示例、PPTX 和测试统一到 heuristic lab 口径 | 确认网站主线服务实用工具，而不是静态展示 |

## 2. 早期工程规范化

2026-04-16 前后，AI 协助完成了仓库规范化：

1. 清理旧架构残留和大体积备份文件。
2. 整理 notebook 命名和目录结构。
3. 修复硬编码路径，改为项目相对路径。
4. 将训练超参数迁移到 `configs/*.yaml`。
5. 编写 README、CLAUDE 文档和初版协同记录。

这部分主要服务开源工程规范，解决“外部用户能否读懂、安装和运行”的问题。
人工侧确认了不把临时交接文件和敏感凭据纳入仓库。

## 3. V13 到 V14：训练失败诊断

V13 GPU 训练出现 reward 发散和 SWAP 上升。AI 协同分析 history、训练日志
和环境代码后，定位出三个主要问题：

- SABRE baseline 在每次 reset 重新计算，吞吐显著下降。
- soft mask 在简单阶段放入过多合法但无益的 SWAP 动作。
- 奖励过度依赖 SABRE 相对值，早期策略容易被大负值压垮。

人工确认后，V14 做了四项修复：

- `sabre_cache.py` 缓存 baseline。
- `env.py` 增加阶段化 action mask。
- 终端奖励按 curriculum stage 分层。
- `pass_manager.py` 改为真正使用 AI trace 构建路由电路。

这一阶段体现了 AI 作为调试助手的作用：先提出可验证假设，再由代码改动和
smoke tests 验证，而不是直接换算法。

## 4. V14 到 V15：算法路线重评估

V14.2 在 GPU 上继续训练后没有稳定收敛，eval avg_swap 仍显著高于 SABRE。
AI 协助并行调研量子路由领域的新路线，重点比较：

- Qiskit SABRE 与 LightSABRE 启发式基线。
- PPO/RL 路由器在复杂拓扑上的训练困难。
- GNN 辅助 MCTS 和 AlphaZero-style 路由方法。

人工确认后，项目没有继续盲目调 PPO 超参数，而是将 V15 定位为
MCTS+GNN 研究型 POC。AI 协助搭建 `src/compiler/v15/` 的网络、树搜索、
self-play、replay buffer 和训练入口，并补充 smoke tests。

后来 P1 真实评测又推翻了“V14 checkpoint 可 warmstart V15”的假设。这一
纠偏被记录到 `docs/technical/decisions.md`，避免继续消耗 GPU 长训。

## 5. P1 评测闭环：拒绝虚假包装

2026-05-04 复查时发现旧 MQT 报告 AI 列为 N/A，无法证明 AI 路由器真的
参与评测。AI 协助完成以下修复：

1. `AIRouter` 支持 CPU 加载 CUDA checkpoint。
2. checkpoint 可从 `model_state` 字段读取 V14 权重。
3. `scripts/eval_mqt_bench.py` 输出 AI 状态、完成率和 outlier 处理。
4. `qcompiler eval` 和 demo 脚本统一显示 `OK`、`INCOMPLETE`、`N/A`。

修复后，P1 主集合结果为：

- SABRE 完成 12/12。
- AI 完成 4/12。
- AI 超越 SABRE 0/4。
- 完成且 SABRE SWAP > 0 的可比行上，AI/SABRE 平均比例为 2.500。

这个结果不支持“AI 已经超过 SABRE”的叙事。AI 协同的关键价值反而是帮助
把项目从过度宣传纠正为真实评测和失败诊断。

## 6. 老师评分主线纠偏

2026-05-05 再次核对原始材料后，确认课题四要求的主线是：

- 开源 Python 编译器插件。
- Qiskit 或 PennyLane 兼容。
- RL/GNN 学习物理拓扑约束。
- 动态初始映射和 SWAP 插入。
- 最小化电路深度和 CNOT/SWAP 数。
- 与 Qiskit SABRE 做性能对比。

AI 协助把 README、项目报告和验收脚本重新对齐到评分表：

- 20% 项目周期管理。
- 25% 开源工程规范。
- 30% 物理机制和算法架构。
- 15% 社区展示。
- 10% AI 协同。

这一轮的人工决策是：不回滚 GitHub Pages，但也不让网页停留在静态展示。
公开网页必须回到算法和工具主线，老师验收仍以算法、插件、SABRE 对比、
报告和可复现命令为核心。

## 7. heuristic lab 与展示材料收口

2026-05-29 到 2026-05-30，AI 协助把项目从旧的 “SABRE vs AI 展示页”
进一步收束为 SabreSwap / LightSABRE heuristic lab。人工确认后的当前口径是：

- 量子信息 / 量子电路大作业要交付真正可用的网站工具，不是静态展示页。
- 算法设计大作业要展示算法选择、实现难度、实验过程、工作量和失败复盘，
  不是把 SABRE 或 GNN 讲成教学课件。
- `lookahead + seed=42 + trials=1` 是当前网站默认口径，因为它快速、稳定，
  且能让静态表和 live API 复现同一组结果。
- bounded search v1 是负样本：`qft10` 上需要 52 个 SWAP，明显差于
  lookahead 的 29 个 SWAP，因此不进入网站默认算法。
- V14/V15 AI checkpoint 只作为历史工作量和失败分析证据，不宣传为胜出算法。

这一阶段的 AI 协同主要体现在一致性检查和证据整理：

1. 将 `docs/index.html`、`src/server/app.py`、public QASM examples 和测试口径
   统一到 `basic`、`lookahead`、`decay` 三种 heuristic。
2. 生成并核查 live API 网站截图，证明网页可以调用本地 API 返回
   `qft10 + lookahead` 的 `SWAP=29`、`depth=156`。
3. 将 bounded search 结果从旧口径修正为当前脚本结果 52 SWAP，并同步到
   PPTX、项目报告和宏观看板。
4. 把算法大作业 PPTX 草稿组织为“问题、候选路线、实验、失败复盘、工程工作量、
   路线图”，而不是教学型算法讲义。
5. 增加 README、public site、API、demo script、experiment script 和 PPTX 的
   contract tests，避免后续改动重新引入虚假宣传。

人工侧保留最终判断权：AI 可以提出页面结构、测试断言和文档草案，但是否把某个
算法纳入网站默认、是否承认失败结果、是否进入最终展示材料，必须由实验数据和
人工决策共同确认。

## 8. AI 协同边界

本项目避免把 AI 用成“代写工具”：

- 关键实验数值必须来自 JSON、Markdown 报告、训练日志或命令输出。
- 文档中的性能结论必须和 `models/v14_tokyo20/eval_report_mqt.*` 一致。
- 网站和 PPTX 中的 heuristic 结论必须和 checked-in QASM examples、API 返回值
  以及测试断言一致。
- 凭据、token、SSH 信息只作为“需要轮换或废弃”的事实处理，不记录值。
- 对外展示不写“AI 已经打败 SABRE”，除非后续评测数据真的支持。

## 9. 总结

AI 在本项目中的高级调用体现在连续诊断和逻辑重构，而不是一次性生成代码。
它帮助项目完成了从 PPO/GNN 路由器、Qiskit 集成、MQT-Bench 评测，到
V15 MCTS/GNN 路线、heuristic lab 网站工具化和失败复盘的完整工程闭环。
最终结果并不完美，但它是可复现、可解释、可继续推进的真实结果。

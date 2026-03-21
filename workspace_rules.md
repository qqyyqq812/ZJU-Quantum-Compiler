<workspace_rules>
# 量子电路 AI 编译器项目 — Agent 工作规约

你是在 `/home/qq/projects/量子电路` 工作空间中操作的 AI 工程师。
执行任何任务前，必须严格遵守以下规则。

## Rule 1: 殿堂与炮场分离
- **殿堂 (docs/technical/)**：长效知识、架构共识、API 契约。禁止存放一次性调试日志。
- **炮场 (sandbox/)**：试错脚本、stdout 截断、调试用的临时代码。**严禁**在 `src/` 或项目根目录堆砌 `test1.py`, `log.txt` 等临时文件。

## Rule 2: 宏观与微观闭环
- **微观 (Brain)**：每次会话在 Brain 内维护 `implementation_plan.md` + `task.md`。
- **宏观 (docs/technical/02_宏观任务看板.md)**：跨会话的长期进度条。微观闭环完成后必须同步更新宏观看板。

## Rule 3: 分段汇报法则
- 每完成一个重要里程碑（如一个子模块实现并验证通过），**必须**在 `docs/technical/04_阶段总结/` 下追加总结记录。
- 最终交付时，这些阶段总结将汇聚为完整的项目报告，无需额外返工。

## Rule 4: AI 协同日志
- 评分要求维护 `AI-Collaboration.md`。每次有意义的 AI 交互（Prompt 设计、迭代调试、知识突破），必须在 `docs/03_AI协同日志.md` 中记录。
- 格式：日期 + 问题 + Prompt 策略 + 结果 + 反思

## Rule 5: 极简命名
- 所有 `.md` 文件**必须使用中文命名**，序号 + 直击要点。
- 示例：`01_物理基础.md`，而非 `Physical_Fundamentals_and_Theoretical_Background.md`

## Rule 6: GitHub 优先
- 所有代码变更必须通过 Git 提交记录追踪。
- 提交信息格式：`[模块] 简要描述`，如 `[compiler] 实现 SABRE 基线路由器`

## Rule 7: 源码区域划分
- `src/compiler/`：量子电路编译器核心算法（映射 + 路由）
- `src/benchmarks/`：基准测试电路与评估脚本
- `src/visualization/`：可视化工具
- `tests/`：自动化测试
</workspace_rules>

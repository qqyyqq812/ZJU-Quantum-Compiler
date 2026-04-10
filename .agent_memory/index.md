# 量子电路 活体脑区入口

> 本脑区通过 MCP 与 Git Hook (post-commit) 实现自动动态化维护。

## 当前状态 (2026-04-10)
- **版本**: V13 (SABRE 相对奖励重构)
- **阶段**: 第三阶段 — 非收敛急救已完成代码修复，待 GPU 训练验证
- **GPU 服务器**: RTX 5090 32GB (AutoDL)
- **关键修改**: env.py / curriculum.py / train.py / gnn_extractor.py / dag.py

## V13 修复清单 (2026-04-10)
1. 奖励函数: SABRE 相对终端奖励 (`sabre_swaps - ai_swaps`)
2. Action Mask: Hard→Soft (允许 delta<=1)
3. GNN 特征: 5维→9维 (增加映射距离、DAG深度、前沿目标距离)
4. 初始映射: 默认随机映射 (消除 Identity 偏见)
5. 课程阈值: 大幅放宽 (匹配 IBM Tokyo 20Q 真实 SABRE 基线)

## 架构路由
- 源码: `src/compiler/` (env, policy, train, curriculum, dag, gnn_*)
- 模型: `models/v13_tokyo20/`
- 文档: `docs/technical/`
- 训练脚本: `run_train_v13.sh`

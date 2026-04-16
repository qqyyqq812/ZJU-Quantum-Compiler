# AI 协同文档

本项目使用 Claude Code 进行 AI 辅助开发，记录 AI 协同的主要任务与方法。

## 协同任务清单

### 项目整改（2026-04-16）

执行完整的项目规范化整改：

1. **清理旧架构残留** - 删除 `.agent_memory/` 知识图谱目录和 `v12_backup.tar.gz`（~42MB）
2. **Jupyter 笔记本规范化** - 清除全部 4 个笔记本的嵌入输出，按 `0N_description.ipynb` 格式重命名并移入 `notebooks/`
3. **硬编码路径修复** - 将 `monitor_dashboard.py` 中的 `/root/projects/量子电路/` 替换为 `os.path.abspath(__file__)` 相对路径；`cleanup_and_setup.sh` 中 `/root/quantum` 参数化为 `$PROJECT_ROOT`
4. **SSH 敏感信息脱敏** - 移除 `run_train_v13.sh` 中的 AutoDL 服务器 IP 和端口
5. **配置管理规范化** - 创建 `configs/` 目录，将 V9/V10/V13 超参数提取为独立 yaml 文件
6. **文档重写** - 重写 `README.md`（含评分对标表）、更新 `CLAUDE.md`、新建本文件

## Prompt 工程实践

### 训练参数管理

通过 `configs/*.yaml` 参数化所有超参数，避免硬编码：

```bash
# 使用配置文件启动训练
python -m src.compiler.train --config configs/v13_gpu.yaml
```

### 实验版本追踪

| 配置文件 | 对应 Notebook | 关键参数 |
|----------|--------------|---------|
| `configs/v9_baseline.yaml` | `01_train_v9_ppo_baseline.ipynb` | rollout_steps=256 |
| `configs/v9_fallback.yaml` | `02_train_v9_ppo_fallback.ipynb` | rollout_steps=2048, soft_mask=true |
| `configs/v13_gpu.yaml` | `run_train_v13.sh` | rollout_steps=32768, GPU |

### 代码审查

使用 Claude Code 进行以下代码审查：
- 验证 RL 训练循环的正确性（环境 reset、rollout 收集、GAE 计算）
- 检查 GNN 编码器的图批处理逻辑
- 审查奖励函数的 SABRE 相对设计合理性

## AI 协同效果评估

| 任务 | 效率提升 | 说明 |
|------|---------|------|
| 项目结构整改 | 大幅提升 | 批量文件重命名、路径修复、配置提取 |
| 文档生成 | 显著 | README、CLAUDE.md 从零生成 |
| 代码审查 | 显著 | 静态分析识别硬编码和安全问题 |
| 调试 | 一般 | RL 训练 bug 仍需人工介入 |

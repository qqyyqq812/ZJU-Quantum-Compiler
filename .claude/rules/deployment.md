# 云端部署协议 (量子电路 RL 项目)

> **覆盖全局规则**：全局 `development-workflow.md` 允许直接改远端，本项目**严禁**在 Colab/GPU 上直接改代码。

## 核心原则：Zero-Touch Remote

**远端（Colab/AutoDL）只做两件事**：
1. `git pull`
2. `bash run_train_v<N>.sh` 或 `run_all cells`

**任何远端代码修改必须通过 Git 流转**：

```
本地修改 → pytest → commit → push → 远端 pull → 运行
         ❌ 禁止：直接 vim 修改远端文件
```

## Colab 部署

### Notebook 约定

- 位置：`notebooks/0N_<desc>.ipynb`
- 只能包含 3 类 cell：
  1. `!git clone` / `!git pull`
  2. `!pip install -r requirements.txt`
  3. `!python -m src.compiler.train --config configs/v<N>_<target>.yaml`
- **禁止**在 notebook 里写算法逻辑、定义函数、覆盖参数
- 提交前用 `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace notebooks/*.ipynb` 清空输出

### 断点续训

```bash
# Colab session 断开后重连，直接:
!python -m src.compiler.train --config configs/v14_gpu.yaml --resume models/v14_tokyo20/checkpoint_ep<last>.pt
```

`train.py` 必须支持 `--resume`（已存在，见 `src/compiler/train.py:155`）。

## AutoDL GPU 部署

### 一次性环境搭建

```bash
# 仅在新实例上跑一次
cd /root && git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git quantum
cd quantum && pip install -r requirements.txt
```

### 常规训练流程

```bash
# 本地：push 最新代码
git push

# 远端 SSH 后：
cd /root/quantum
git pull
nohup bash run_train_v14.sh > /dev/null 2>&1 &
echo $!  # 记录 PID
```

### 监控约定

- 训练日志：`models/v<N>_tokyo20/training_v<N>.log`
- GPU 利用率监控：`nvidia-smi -l 5`
- 训练存活探针：`ps -p <PID>` 或查看 `models/v<N>_tokyo20/history_*.json` 的修改时间

## 关机纪律

**训练完毕或任何中断后，立即关机**：

- AutoDL 不关机 = 持续计费
- Colab 不手动 `Stop` = 占用 GPU quota
- 每次 SSH 结束前，**必须**检查 `ps aux | grep train` 并清理遗留进程

## Secrets 管理

- **禁止**把 SSH 密码、API Key 写入任何文件（包括 README、notebook、script）
- SSH 连接信息放 `~/.ssh/config`，不进 Git
- 提交前用 `git diff --cached | grep -iE 'password|secret|token|key'` 检查

## 版本一致性校验

部署前检查本地和远端代码是否一致：

```bash
# 远端
cd /root/quantum && git log -1 --oneline

# 本地
git log -1 --oneline

# 两者的 commit hash 必须一致
```

## 失败恢复

| 症状 | 处理 |
|-----|------|
| GPU segfault | 先查 `torch_geometric` 是否被拉入 — 本项目已纯 PyTorch，应该没有 |
| 训练卡住不晋级 | 查 `final_info` 是否正常提取（V13 已修 bug, see 3919059） |
| 磁盘满 | `bash cleanup_and_setup.sh`（见根目录） |
| 断网 | Colab 自动 checkpoint，AutoDL 用 `--resume` 续跑 |

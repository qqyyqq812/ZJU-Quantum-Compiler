# 强化学习量子电路路由训练方法 — 深度研究报告
> 🔍 搜索轮次：1 轮（定向深搜） | 覆盖主题：Action Masking, Reward Shaping, Tabu Search | 生成时间：2026-04-02

## 核心发现
- **Action Masking (动作掩码) 是绝对标配**：在处理高维约束空间（如量子路由的拓扑图）时，“无效动作掩码”被广泛用于阻止 Agent 做出非法语义（含死循环/Tabu）。这大大加速了训练并通过把模型强行拉在可行解空间来提升策略质量。
- **依赖体罚的 Anti-Pattern**：单纯使用高额 Penalty（如给 Tabu 动作设 `-5.0` 惩罚）来替代物理掩码，在基于 PPO 的量子编译模型中是已经被证明会导致“奖励稀疏/淹没”从而无法收敛的通病。
- **混合与前瞻奖励**：当前的 SOTA (State of the Art) 模型倾向于使用 Hybrid Rewards，将静态/领域知识（如前沿门的物理距离变化）与明确的终端疏通奖励结合。

## 详细分析
### 1. 动作掩码 (Action Masking) vs 惩罚 (Penalty)
现有顶会的实现（如基于 ZX-calculus 优化或 PPO 路由）明确指出，针对网络难以通过试错学会的严苛死规（如左右反复横跳、无效的 SWAP 操作），最好的实践是直接在数学层面将该动作的概率 Logit 设为 $-\infty$（即物理掩码）。这不仅避免了 PPO 的价值网络 $V(s)$ 发散，还极大节省了采样成本。
如果把 Tabu 交由 `-5` 这种 Penalty 去处理，意味着它每走错一步就在浪费一次 Rollout 收集，最终必然陷入僵局死锁，符合当前 `v9` 模型由于大量无意义的试探导致 `episode_swaps` 暴涨至 244 且梯度彻底停滞（P_Loss: -0.0001）。

### 2. Tabu Search的借用机制
量子编译中经常借用 Tabu（禁忌表）的思想，但文献表明，Tabu 主要被用来作**决定性的路径阻隔**（Deterministic local optima avoidance），而不是惩罚项。即一旦记录了最近的 $N$ 步操作，这 $N$ 步的反向操作将彻底从合法动作集中移除。

### 3. Reward Shaping 设计共识
- **SWAP Cost**：标准的负向奖励，每次执行一个产生串扰或门开销的 SWAP 给定一定负激励（如 `-0.5`）。
- **Sparse Positive Excitement**：必须要在电路进度推进（如某些门可无障碍在物理硬件上执行）时给予正向刺激，从而抵消 SWAP 的连续扣分，拉动梯度。

## 与我们的 V10 方案（急诊抢救）的印证
1. **恢复 Tabu 的物理硬掩码**：这与调研结果高度一致。取消物理掩码让模型自由翻滚是违背当前前沿训练共识的，必须在 `get_action_mask` 中将 `tabu_list` 涵盖的边直接封杀。
2. **简化奖励、破除惩罚围城**：取消或者极大弱化 `penalty_tabu` 等微观干预惩罚，符合现代 RL 环境设计的“Keep It Simple and Physical”原则。

## 参考来源
1. Qubit Mapping Research Contexts (TU Delft, arxiv): PPO implementations for qubit routing constraint handling.
2. Invalid Action Masking in RL: Standard practice for pruning illegal circuit transformations.

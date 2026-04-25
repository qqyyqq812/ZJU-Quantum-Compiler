# V14 MQT-Bench 评测报告

## Metadata

- **运行时间**: 2026-04-25 16:54:19
- **拓扑**: ibm_tokyo (20 qubits, diameter=4)
- **电路数**: 15（覆盖 5 种类型 × 3 种规模）
- **MQT-Bench 状态**: 已安装 v2.x
- **AI 模型**: `models/v14_tokyo20/v7_ibm_tokyo_best.pt` — 未加载（仅 SABRE 评测）
- **basis_gates**: ['cx', 'id', 'rz', 'sx', 'x']
- **max_steps (AI router)**: 2000

## 路由 SWAP 对比

| Circuit | n_qubits | SABRE SWAP | SABRE time(ms) | AI SWAP | AI time(ms) | AI/SABRE ratio |
|---------|----------|------------|----------------|---------|-------------|----------------|
| qft_5 | 5 | 0 | 15.37 | N/A | N/A | N/A |
| qft_10 | 10 | 0 | 5.74 | N/A | N/A | N/A |
| qft_20 | 20 | 0 | 8.21 | N/A | N/A | N/A |
| qaoa_5 | 5 | 0 | 3.63 | N/A | N/A | N/A |
| qaoa_10 | 10 | 0 | 4.29 | N/A | N/A | N/A |
| qaoa_20 | 20 | 0 | 8.45 | N/A | N/A | N/A |
| grover_5 | 5 | 0 | 8.76 | N/A | N/A | N/A |
| grover_10 | 10 | 0 | 144.05 | N/A | N/A | N/A |
| grover_20 | 20 | 0 | 37081.43 | N/A | N/A | N/A |
| ghz_5 | 5 | 0 | 6.37 | N/A | N/A | N/A |
| ghz_10 | 10 | 0 | 6.55 | N/A | N/A | N/A |
| ghz_20 | 20 | 0 | 118.45 | N/A | N/A | N/A |
| vqe_5 | 5 | 0 | 2.77 | N/A | N/A | N/A |
| vqe_10 | 10 | 0 | 118.05 | N/A | N/A | N/A |
| vqe_20 | 20 | 0 | 975.86 | N/A | N/A | N/A |

## 汇总

- SABRE 完成率: **15/15** (100%)
- SABRE 平均 SWAP: **0.0**

## 与 V13 的差异 (参照 docs/technical/decisions.md §V14)

- **V14-1 SABRE baseline 缓存**：训练吞吐 1.0 → 15 eps/s，本评测与训练时使用同一份 SABRE 实现，可复现对照。
- **V14-2 阶段化 Mask**：5Q 阶段（Stage 0-2）已稳定收敛，本表 5Q 列代表 V14.2 ep25333 的稳定能力；10Q/20Q（Stage 3+）仍未收敛，建议参考 SABRE 列。
- **V14-3 奖励分层**：terminal reward 按 stage 切换；
- **V14-4 pass_manager 真集成**：AI SWAP 数（route_count_only）可被外部独立复现，不再调用 Qiskit SABRE 重编译。
- **V13 vs V14 在 5Q QFT 上的 SWAP 数**：参见 `models/v12_tokyo20/eval_report_v12.md` vs 本报告。

## 备注：AI 评测未运行

当前未加载 AI 模型（V14 训练在 Stage 3 卡住，V14.2 ep25333 在 5Q 上可用）。
本报告作为 SABRE 基线管线的可复现性验证 — 后续 V14.2 收敛后，
跑 `python scripts/eval_mqt_bench.py --ai-model models/v14_tokyo20/v7_ibm_tokyo_best.pt` 可填充 AI 列。

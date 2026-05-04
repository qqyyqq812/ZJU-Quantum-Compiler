# V14 MQT-Bench 评测报告

## Metadata

- **运行时间**: 2026-05-04 22:20:00
- **拓扑**: ibm_tokyo (20 qubits, diameter=4)
- **电路数**: 12（覆盖 4 种类型 × 3 种规模）
- **MQT-Bench 状态**: 已安装 v2.x
- **AI 模型**: `models/v14_tokyo20/checkpoint_ep25333.pt` — 已加载
- **basis_gates**: ['cx', 'id', 'rz', 'sx', 'x']
- **max_steps (AI router)**: 2000

## 路由 SWAP 对比

| Circuit | n_qubits | SABRE SWAP | SABRE time(ms) | AI status | AI SWAP | AI gates | AI time(ms) | AI/SABRE ratio |
|---------|----------|------------|----------------|-----------|---------|----------|-------------|----------------|
| qft_5 | 5 | 8 | 9.73 | INCOMPLETE | 3 | 36 | 20563.42 | 0.375 |
| qft_10 | 10 | 30 | 6.99 | INCOMPLETE | 25 | 137 | 31726.16 | 0.833 |
| qft_20 | 20 | 163 | 15.53 | INCOMPLETE | 4 | 37 | 98292.72 | 0.025 |
| qaoa_5 | 5 | 9 | 19.87 | INCOMPLETE | 8 | 28 | 21635.06 | 0.889 |
| qaoa_10 | 10 | 34 | 6.89 | INCOMPLETE | 11 | 53 | 32941.21 | 0.324 |
| qaoa_20 | 20 | 174 | 18.51 | INCOMPLETE | 313 | 741 | 30247.46 | 1.799 |
| ghz_5 | 5 | 0 | 5.18 | OK | 0 | 7 | 63.37 | 1.000 |
| ghz_10 | 10 | 3 | 6.30 | OK | 6 | 12 | 128.39 | 2.000 |
| ghz_20 | 20 | 10 | 5.15 | INCOMPLETE | 3 | 6 | 17669.79 | 0.300 |
| vqe_5 | 5 | 0 | 7.75 | OK | 0 | 112 | 363.57 | 1.000 |
| vqe_10 | 10 | 4 | 12.23 | OK | 12 | 227 | 602.75 | 3.000 |
| vqe_20 | 20 | 24 | 16.00 | INCOMPLETE | 49 | 417 | 25351.80 | 2.042 |

## 汇总

- 主集合电路数: **12/12**
- SABRE 完成率: **12/12** (100%)
- SABRE 平均 SWAP（排除 outlier）: **38.2**
- AI 完成率: **4/12** (33%)
- AI 超越 SABRE: **0/4** (0%)
- AI/SABRE 平均比例（仅完成且 SABRE>0）: **2.500**

## 与 V13 的差异 (参照 docs/technical/decisions.md §V14)

- **V14-1 SABRE baseline 缓存**：训练吞吐 1.0 → 15 eps/s，本评测与训练时使用同一份 SABRE 实现，可复现对照。
- **V14-2 阶段化 Mask**：本次 P1 评测没有支持“5Q 已稳定可用”的旧假设；`qft_5` 与 `qaoa_5` 均未完成。
- **V14-3 奖励分层**：terminal reward 按 stage 切换；
- **V14-4 pass_manager 真集成**：AI SWAP 数（route_count_only）可被外部独立复现，不再调用 Qiskit SABRE 重编译。
- **V14 ep25333 战力结论**：12 条主集合电路中 AI 只完成 4 条；完成且 SABRE>0 的 2 条可比电路平均 AI/SABRE = 2.500，暂不能作为 V15 warmstart 的可靠证据。

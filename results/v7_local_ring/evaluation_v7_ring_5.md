# V7 Quantum Router Evaluation Report

- **Model**: `models/v7_local/v7_linear_5_best.pt`
- **Topology**: `ring_5` (5 qubits)
- **Trials per circuit**: 3
- **Baseline**: Qiskit SABRE (O3)

## Benchmark Results

| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |
|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|
| qft_5 | 5 | 10 | 999.0 | 9.0 | 0.01x | 360.8 | 8.1 |
| grover_5 | 5 | 19 | 999.0 | 4.0 | 0.00x | 374.3 | 9.8 |
| qaoa_5 | 5 | 17 | 0.0 | 0.0 | **1.00x** | 29.8 | 5.0 |
| random_5 | 5 | 10 | 7.0 | 0.0 | 0.00x | 21.8 | 8.2 |
| **Overall (GeoMean)** | - | - | - | - | **0.00x** | - | - |

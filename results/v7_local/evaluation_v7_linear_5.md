# V7 Quantum Router Evaluation Report

- **Model**: `models/v7_local/v7_linear_5_best.pt`
- **Topology**: `linear_5` (5 qubits)
- **Trials per circuit**: 3
- **Baseline**: Qiskit SABRE (O3)

## Benchmark Results

| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |
|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|
| qft_5 | 5 | 10 | 22.0 | 10.0 | 0.45x | 33.1 | 7.9 |
| grover_5 | 5 | 19 | 12.0 | 4.0 | 0.33x | 50.9 | 7.6 |
| qaoa_5 | 5 | 17 | 3.0 | 2.0 | 0.67x | 28.6 | 8.3 |
| random_5 | 5 | 10 | 14.0 | 5.0 | 0.36x | 32.0 | 7.4 |
| **Overall (GeoMean)** | - | - | - | - | **0.44x** | - | - |

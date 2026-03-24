# V7 Quantum Router Evaluation Report

- **Model**: `models/v7_linear_5.pt`
- **Topology**: `ring_5` (5 qubits)
- **Trials per circuit**: 2
- **Baseline**: Qiskit SABRE (O3)

## Benchmark Results

| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |
|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|
| qft_5 | 5 | 10 | 999.0 | 10.5 | 0.01x | 906.4 | 23.4 |
| grover_5 | 5 | 19 | 999.0 | 4.0 | 0.00x | 799.2 | 10.6 |
| qaoa_5 | 5 | 17 | 0.0 | 0.0 | **1.00x** | 49.8 | 7.4 |
| random_5 | 5 | 10 | 999.0 | 0.0 | 0.00x | 741.6 | 13.5 |
| **Overall (GeoMean)** | - | - | - | - | **0.00x** | - | - |

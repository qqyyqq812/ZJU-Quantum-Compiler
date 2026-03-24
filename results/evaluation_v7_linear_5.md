# V7 Quantum Router Evaluation Report

- **Model**: `models/v7_linear_5.pt`
- **Topology**: `linear_5` (5 qubits)
- **Trials per circuit**: 1
- **Baseline**: Qiskit SABRE (O3)

## Benchmark Results

| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |
|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|
| qft_5 | 5 | 10 | 999.0 | 10.0 | 0.01x | 698.7 | 65.7 |
| grover_5 | 5 | 19 | 999.0 | 4.0 | 0.00x | 776.0 | 13.2 |
| qaoa_5 | 5 | 17 | 999.0 | 2.0 | 0.00x | 457.1 | 11.1 |
| random_5 | 5 | 10 | 999.0 | 5.0 | 0.01x | 729.6 | 10.6 |
| **Overall (GeoMean)** | - | - | - | - | **0.00x** | - | - |

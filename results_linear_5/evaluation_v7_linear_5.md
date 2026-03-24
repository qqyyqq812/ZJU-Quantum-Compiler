# V7 Quantum Router Evaluation Report

- **Model**: `models/v7_linear_5.pt`
- **Topology**: `linear_5` (5 qubits)
- **Trials per circuit**: 2
- **Baseline**: Qiskit SABRE (O3)

## Benchmark Results

| Circuit | Qubits | Depth | AI SWAPs (Med) | SABRE SWAPs (Med) | **SWAP Ratio** | AI Time (ms) | SABRE Time (ms) |
|---------|--------|-------|----------------|-------------------|----------------|--------------|-----------------|
| qft_5 | 5 | 10 | 999.0 | 10.0 | 0.01x | 429.4 | 25.0 |
| grover_5 | 5 | 19 | 999.0 | 4.0 | 0.00x | 627.0 | 26.3 |
| qaoa_5 | 5 | 17 | 999.0 | 2.0 | 0.00x | 534.8 | 26.9 |
| random_5 | 5 | 10 | 999.0 | 5.0 | 0.01x | 521.0 | 11.1 |
| **Overall (GeoMean)** | - | - | - | - | **0.00x** | - | - |

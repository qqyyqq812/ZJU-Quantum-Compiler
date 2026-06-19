# Final submission summary

This document summarizes the public evidence and demonstration path for the
quantum circuit routing compiler. The repository presents a complete loop:
OpenQASM input, hardware topology selection, NPQR routing, SABRE basic
comparison, SWAP trace replay, REST access, MCP access, and a browser
playground.

## Result scope

The main reported result uses representative 10/20-qubit routing tasks. NPQR
completed all rows, returned replayable traces, and reduced SWAP count relative
to SABRE basic on the selected benchmark set.

| case | qubits | NPQR swaps | SABRE basic swaps | delta |
| --- | ---: | ---: | ---: | ---: |
| GHZ10 | 10 | 2 | 6 | -4 |
| QFT10 | 10 | 6 | 13 | -7 |
| QAOA10 | 10 | 0 | 15 | -15 |
| VQE10 | 10 | 1 | 4 | -3 |
| Random10 | 10 | 17 | 19 | -2 |
| Brickwork20 | 20 | 15 | 31 | -16 |
| CliqueBlocks20 | 20 | 19 | 38 | -19 |
| DeepRandom20 | 20 | 105 | 135 | -30 |
| RingEntangler20 | 20 | 16 | 41 | -25 |
| SparseRandom20 | 20 | 37 | 57 | -20 |

The selected 30/50-qubit examples are extension-scale demonstrations. They are
intended for a deployed backend, while the static GitHub Pages page remains the
fastest entry point for 5/10-qubit browser review.

| case | qubits | NPQR swaps | SABRE basic swaps | delta |
| --- | ---: | ---: | ---: | ---: |
| LineGHZ30 | 30 | 23 | 43 | -20 |
| Random30-d4 | 30 | 61 | 67 | -6 |
| LineGHZ50 | 50 | 50 | 79 | -29 |
| RingSparse50 | 50 | 98 | 113 | -15 |

## Public entry points

The browser playground is the human-facing entry point. It shows QASM input,
topology choice, route trace replay, SWAP comparison, depth comparison, and
NPQR/SABRE results.

The REST API is the real compiler backend for the playground and scripts. It
serves validation, examples, topology data, compilation, and benchmark evidence.

The MCP service exposes the same compiler capabilities to tool clients and
automation workflows. It is useful for scripted checks and agent-assisted
experiments, but it is not required for normal browser use.

## Demonstration path

Start with `ghz5`, `qft5`, or `qaoa5` to inspect the page layout and trace
replay quickly. Then run a 10-qubit example, compare the NPQR and SABRE basic
metrics, and inspect how SWAP insertion changes the physical route. With a
backend running, use `line_ghz30` or `line_ghz50` to demonstrate the selected
extension-scale path.

The same evidence appears through the browser, REST response, MCP evidence
tool, checked-in examples, and the report appendix.

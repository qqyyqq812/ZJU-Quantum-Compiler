# NPQR final closure report

This report summarizes the final closure target for NPQR. The project goal is
not to claim state-of-the-art quantum routing. The goal is to
show a complete, explainable, and reproducible neural-assisted routing system
that beats the SABRE basic baseline on representative 10/20-qubit circuits and
shows useful scaling behavior on selected 30/50-qubit bounded tests. A final
bounded 80/100-qubit test is used only to define the current scale boundary.

## Final claim

NPQR combines neural action scoring, initial mapping selection, bounded search,
frontier pruning, suffix repair, and trace replay. SABRE is used only as a
comparison baseline and is not used as an NPQR fallback.

The final project claim is:

- NPQR completes the representative 10/20-qubit benchmark set.
- NPQR beats SABRE basic on the representative 10/20-qubit benchmark set.
- NPQR results are replayable and use no SABRE fallback.
- NPQR can run on selected 30/50-qubit circuits and wins on several larger
  structured cases.
- The largest currently demonstrated completed scale is 50 qubits.
- NPQR is slower than SABRE and does not win every large-scale case.

## 10/20-qubit basic baseline result

The representative 10/20-qubit result is the main performance evidence for the
project.

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

Summary:

- Completed: `10/10`
- Beats SABRE basic: `10/10`
- SABRE fallback: `0`
- Trace replay: passed for completed NPQR rows

## 30/50-qubit scale test

The scale test checks whether the current NPQR runtime has useful behavior
beyond 20 qubits. It is not a full benchmark.

| case | qubits | NPQR swaps | SABRE basic swaps | delta |
| --- | ---: | ---: | ---: | ---: |
| LineGHZ30 | 30 | 23 | 43 | -20 |
| Random30-d4 | 30 | 61 | 67 | -6 |
| LineGHZ50 | 50 | 50 | 79 | -29 |
| RingSparse50 | 50 | 98 | 113 | -15 |

Summary:

- Completed: `4/4`
- Beats SABRE basic: `4/4`
- SABRE fallback: `0`

The scale test supports the claim that NPQR has useful larger-scale potential.
It also shows that the final system has a demonstrated 50-qubit
completion range for selected structures.

## 80/100-qubit boundary test

The final large-scale test also tried to push beyond the demonstrated range.
This part defines the boundary and is not a success claim.

| case | qubits | NPQR result | SABRE basic swaps | conclusion |
| --- | ---: | --- | ---: | --- |
| LineGHZ80 | 80 | timeout at 240s | 137 | not completed in CPU budget |
| Random80-d2 | 80 | stopped before completion | - | not claimed |
| LineGHZ100 | 100 | not run | - | not claimed |
| RingSparse100 | 100 | not run | - | not claimed |

The current honest upper bound is therefore 50 qubits. The 80/100-qubit rows
are useful as optimization guidance, not as final project capability claims.

## Recommended demo path

Use a short demo that shows one small, one 20-qubit, and one larger-scale case.

1. Show `qaoa_10` or `qft_10` for a compact 10-qubit win.
2. Show `brickwork_20` or `deep_random_20` for the 20-qubit result.
3. Show `line_ghz_50` or `ring_sparse_50` for scaling potential.
4. Mention `line_ghz_80` as the honest CPU-bounded scale boundary.

## Closure boundary

The project is ready to close when the report, website, and public tests all
use the same claim:

> NPQR beats SABRE basic on representative 10/20-qubit routing tasks, returns
> replayable no-fallback traces, and shows useful but not universal scaling
> behavior up to selected 50-qubit bounded tests.

Do not claim that NPQR beats SABRE lookahead across the board. Do not claim
state-of-the-art routing. Do not start GPU training for the final submission.

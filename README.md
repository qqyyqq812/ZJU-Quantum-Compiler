# ZJU Quantum Compiler

A neural-assisted quantum routing compiler for constrained quantum hardware.
The default route is NPQR: a learned routing policy combined with initial
mapping selection, bounded beam search, pruning, and local suffix repair.
SABRE is included as the comparison baseline, not as a hidden fallback.

## What this project does

Real quantum chips do not let every qubit interact with every other qubit.
A compiler must map logical qubits onto a hardware coupling graph and insert
SWAP gates so every two-qubit gate becomes physically executable. This project
turns that routing problem into a graph search task and exposes it through a
Python package, REST API, and HTTP MCP server.

The public repository contains the final runnable version:

- `src/`: compiler, NPQR runtime, REST API, and MCP server.
- `examples/`: OpenQASM circuits for quick demonstrations.
- `models/default/npqr-default.pt`: the default NPQR inference model.
- `docs/`: website, presentation material, and team report guidance.
- `scripts/`: final readiness, packaging, and algorithm matrix utilities.
- `tests/`: public API, MCP, README, deployment, and readiness checks.

## Install

Use Python 3.10 or newer.

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Check the package and model status:

```bash
qcompiler info
```

## Run the REST API

The REST API is the easiest way to connect a web page or script.

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8765
```

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Project status, default backend, model path, and load status. |
| `GET /api/examples` | Checked-in OpenQASM examples. |
| `POST /api/compile` | Compile an example or inline OpenQASM with NPQR or SABRE. |
| `GET /api/benchmarks` | Public algorithm boundary and final smoke summary. |
| `GET /api/npqr/evidence` | Machine-readable NPQR algorithm description. |

Compile a checked-in example with the default NPQR route:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

Compile inline OpenQASM:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"qasm":"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"}'
```

The NPQR response includes routed QASM, SWAP count, depth, elapsed time, a
route trace, and a SABRE baseline field for comparison. The `components` field
marks that `sabre_fallback` is false.

## Run the MCP server

The MCP server exposes the same compiler as callable tools.

```bash
qcompiler-mcp-http
```

By default the MCP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Main MCP tools:

| Tool | Purpose |
| --- | --- |
| `qcompiler_status` | Report default backend, model, and algorithm boundary. |
| `list_examples` | List checked-in QASM examples. |
| `compile_npqr` | Compile an example with NPQR. |
| `compile_sabre` | Compile an example with SABRE. |
| `compile_qasm` | Compile user-provided OpenQASM 2 text. |
| `get_benchmarks` | Return public benchmark and claim boundaries. |
| `get_npqr_boundary` | Explain what NPQR claims and does not claim. |
| `get_algorithm_evidence` | Return the course-facing algorithm component summary. |

## Algorithm summary

NPQR is a combination pipeline rather than a single black-box neural network.
The neural model is the core action scorer, while initial mapping selection and
other classical search components keep the route valid and controllable.

1. Parse the input OpenQASM circuit and the IBM Tokyo coupling graph.
2. Build dependency information for executable front gates.
3. Generate candidate logical-to-physical initial mappings.
4. Use the neural policy to score valid SWAP actions in each state.
5. Run bounded beam search so several candidate routes survive each step.
6. Trigger stronger frontier search only on difficult interaction patterns.
7. Prune weak actions to control runtime.
8. Repair short unfinished suffixes with bounded local search.
9. Replay the final trace and emit topology-valid routed QASM.

SABRE is the baseline. It is a strong Qiskit heuristic that chooses SWAPs from
front-layer and lookahead distance estimates. In this project SABRE is used for
comparison metrics and explanation, not as the self-developed algorithm and not
as a fallback that completes NPQR routes.

## Course algorithm mapping

The project can be explained with standard algorithm-design concepts:

| Course concept | How it appears here |
| --- | --- |
| Graph problem | The chip is a graph; qubits are vertices and couplings are edges. |
| Transform-and-conquer | Circuit routing becomes mapping plus graph-constrained path planning. |
| Greedy heuristic | Front-gate distance reduction guides local SWAP quality. |
| Decrease-and-conquer | Every executed gate reduces the remaining routing task. |
| Time-space tradeoff | Distance matrices, candidate mappings, and beams use memory to reduce repeated work. |
| Iterative improvement | Mapping and route quality improve through SWAP updates and suffix repair. |
| Search pruning | Beam width, action pruning, and trigger rules limit the search tree. |
| Approximation | The route is a bounded high-quality solution, not a proof of global optimum. |
| Neural network | The model provides learned action preferences inside the search loop. |

## Examples

Checked-in examples cover small and medium routing pressure:

| Example | Description |
| --- | --- |
| `examples/ghz5.qasm` | Small entanglement chain. |
| `examples/qft5.qasm` | Compact dense interaction pattern. |
| `examples/qaoa5.qasm` | Small QAOA-style layer. |
| `examples/qft10.qasm` | Larger dense routing pressure. |
| `examples/qaoa10.qasm` | Medium optimization-style circuit. |
| `examples/ghz10.qasm` | Larger entanglement chain. |
| `examples/vqe10.qasm` | VQE-like ansatz. |

Run the final quick matrix:

```bash
python scripts/experiment_algorithm_matrix.py --quick
```

Generate JSON for report tables:

```bash
python scripts/experiment_algorithm_matrix.py --quick --json
```

## Website and report material

`docs/index.html` is the GitHub Pages entry page. It keeps the visual interface
stable and calls the REST API for live compilation. The team-facing report guide
is in `docs/plans/2026-06-05-mcp-work-split.md`; it explains the final algorithm
without internal experiment names.

Use `docs/playground-user-guide.md` when you need to explain the web page to a
reviewer or teammate. It covers **Run**, **Step**, **Reset**, Examples, Custom
QASM, Generate, NPQR versus the fixed SABRE baseline, Tokyo mapping, route
trace, `compiled_qasm`, REST API, and the advanced MCP boundary.

Presentation material is stored under `docs/slides/`.

## Deployment

REST API Docker image:

```bash
docker build -f Dockerfile.api -t zju-quantum-compiler-api .
docker run --rm -p 8080:8080 zju-quantum-compiler-api
```

MCP Docker image:

```bash
docker build -f Dockerfile.mcp -t zju-quantum-compiler-mcp .
docker run --rm -p 8081:8081 zju-quantum-compiler-mcp
```

Render blueprints are provided in `render.yaml` and `render-mcp.yaml`.

## Readiness checks

Run the public contract tests before publishing:

```bash
python -m pytest -q \
  tests/test_public_api.py \
  tests/test_public_site.py \
  tests/test_public_deploy_config.py \
  tests/test_readme_contract.py \
  tests/test_submission_readiness.py \
  tests/test_algorithm_matrix.py \
  tests/test_demo_scripts.py
python scripts/check_submission_readiness.py
git diff --check
```

Build a local review package:

```bash
python scripts/package_submission.py
```

Generated review files are written under `results/submission_package/` and are
not committed.

## Boundaries

This repository intentionally keeps the public tree small. It includes the final
runtime, one default model, examples, deployment files, report guidance, and
public tests. Internal training sweeps, temporary handoff notes, exploratory
work files, and intermediate results are ignored from the public release branch.

The honest claim is: NPQR is a runnable neural-assisted routing workflow with a
clear API/MCP surface and course-reportable algorithm design. It is not claimed
to dominate SABRE on every circuit.

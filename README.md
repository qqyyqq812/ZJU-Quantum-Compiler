# ZJU Quantum Compiler

ZJU Quantum Compiler is a neural-assisted quantum circuit routing compiler for
restricted hardware topologies. It accepts OpenQASM 2 circuits, maps logical
qubits to physical qubits, inserts the required SWAP gates, and returns a
routed circuit that satisfies the target coupling graph.

The default compiler path is NPQR. NPQR combines neural action scoring with
initial mapping selection, bounded beam search, action pruning, local suffix
repair, and trace replay validation. SABRE is kept as a fixed Qiskit baseline
for reproducible comparison and is not used as a hidden fallback for NPQR.

## Online demo

The public browser console is available at:

```text
https://qqyyqq812.github.io/ZJU-Quantum-Compiler/
```

The page is served from `docs/index.html`. It can call the deployed REST API
when the API is reachable, and it also documents the MCP endpoint in the folded
service panel. The user guide is available at
`docs/playground-user-guide.md`.

## Repository contents

The repository keeps the runnable compiler, browser console, public examples,
and technical report materials in one place:

- `src/`: NPQR runtime, compiler utilities, REST API, and MCP service.
- `examples/`: checked-in OpenQASM examples for 5, 10, 30, and 50 qubits.
- `models/default/npqr-default.pt`: default NPQR inference checkpoint.
- `docs/index.html`: the GitHub Pages compiler console.
- `docs/项目说明.md`: detailed Chinese technical guide.
- `docs/ai-collaboration.md`: concise AI collaboration disclosure.
- `docs/report_latex/main.pdf`: technical report PDF.
- `scripts/`: reproducibility, readiness, and packaging scripts.
- `tests/`: public API, site, documentation, and release contract tests.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Check the installed command and default model:

```bash
qcompiler info
```

## REST API

Start the FastAPI service for browser, script, or backend calls:

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8765
```

The public REST surface is:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Return backend, model, and default compiler status. |
| `GET /api/examples` | List checked-in OpenQASM examples and default topologies. |
| `POST /api/validate` | Validate inline OpenQASM against a selected topology. |
| `POST /api/compile` | Compile a checked-in example or inline QASM. |
| `POST /api/compile/jobs` | Create an asynchronous compile job with real phase timing. |
| `GET /api/compile/jobs/{job_id}` | Poll an asynchronous compile job. |
| `GET /api/topology/{name}` | Return topology metadata and JSON-safe edges. |
| `GET /api/benchmarks` | Return public benchmark and boundary summaries. |
| `GET /api/npqr/evidence` | Return machine-readable NPQR evidence. |

Compile a small checked-in example with NPQR:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

Compile a 50-qubit checked-in example on the 5x10 grid with SABRE:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"line_ghz50","backend":"sabre","topology":"grid_5x10"}'
```

Compile custom OpenQASM:

```bash
curl -s http://127.0.0.1:8765/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"qasm":"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"}'
```

Compile responses include routed QASM, SWAP count, depth, elapsed time,
route trace events, initial and final mappings, and the SABRE comparison
baseline. The `components.sabre_fallback` flag is `false` for NPQR results.

## MCP service

Start the HTTP MCP service for tool clients:

```bash
qcompiler-mcp-http
```

The default endpoint is:

```text
http://127.0.0.1:8000/mcp
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

The MCP tools expose the same read-only compiler capabilities:

| Tool | Purpose |
| --- | --- |
| `qcompiler_status` | Return compiler, model, and boundary status. |
| `list_examples` | List checked-in QASM examples. |
| `compile_npqr` | Compile a checked-in example with NPQR. |
| `compile_sabre` | Compile a checked-in example with SABRE. |
| `compile_qasm` | Compile user-provided OpenQASM 2 text. |
| `get_benchmarks` | Return public benchmark summaries. |
| `get_npqr_boundary` | Return explicit NPQR claims and non-claims. |
| `get_algorithm_evidence` | Return algorithm component evidence. |

## Command line

Inspect the installation:

```bash
qcompiler info
```

Compile a checked-in example:

```bash
qcompiler compile --example qft5 --backend npqr
```

Compile a QASM file:

```bash
qcompiler compile examples/qft5.qasm --backend sabre --topology tokyo
```

Generate the reproducible comparison matrix:

```bash
qcompiler matrix --quick
```

## Examples and topologies

Small and medium examples use IBM Tokyo 20Q. Larger examples use selected grid
topologies that match their input size.

| Example | Qubits | Default topology | File |
| --- | ---: | --- | --- |
| QFT 5 | 5 | `tokyo` | `examples/qft5.qasm` |
| GHZ 5 | 5 | `tokyo` | `examples/ghz5.qasm` |
| QAOA 5 | 5 | `tokyo` | `examples/qaoa5.qasm` |
| QFT 10 | 10 | `tokyo` | `examples/qft10.qasm` |
| QAOA 10 | 10 | `tokyo` | `examples/qaoa10.qasm` |
| GHZ 10 | 10 | `tokyo` | `examples/ghz10.qasm` |
| VQE-like 10 | 10 | `tokyo` | `examples/vqe10.qasm` |
| LineGHZ30 | 30 | `grid_5x6` | `examples/line_ghz30.qasm` |
| Random30-d4 | 30 | `grid_5x6` | `examples/random30_d4.qasm` |
| LineGHZ50 | 50 | `grid_5x10` | `examples/line_ghz50.qasm` |
| RingSparse50 | 50 | `grid_5x10` | `examples/ring_sparse50.qasm` |

Supported topology aliases include `tokyo`, `grid_5x6`, and `grid_5x10`.
Tokyo has 20 physical qubits. A 30-qubit circuit must use `grid_5x6`, and a
50-qubit circuit must use `grid_5x10`.

## Algorithm overview

Quantum routing is a graph-constrained optimization problem. Physical qubits
are graph vertices, and hardware couplings are graph edges. A two-qubit gate can
run directly only when the two mapped physical qubits are adjacent. Otherwise,
the compiler inserts SWAP gates to move logical states across valid edges.

NPQR follows this pipeline:

1. Parse the OpenQASM circuit and selected hardware topology.
2. Build gate dependencies and identify executable frontier gates.
3. Generate candidate logical-to-physical initial mappings.
4. Score legal SWAP actions with a neural model.
5. Keep multiple route candidates with bounded beam search.
6. Apply trigger-gated frontier search on difficult interaction patterns.
7. Prune low-value actions to control runtime and candidate growth.
8. Repair difficult suffixes with bounded local search.
9. Replay the selected trace and emit QASM only after topology validation.

The design combines standard algorithmic ideas: graph modeling, problem
transformation, greedy local scoring, decremental progress through executed
gates, space-time tradeoffs in cached distances and candidate states, iterative
improvement through SWAPs, search pruning, and bounded approximate solving.

## Results and boundaries

The public evidence uses SABRE basic as the fixed quality baseline. The final
local evaluation shows that NPQR completes the representative 10/20-qubit set
and returns fewer SWAP gates than SABRE basic on all listed rows. The selected
30/50-qubit examples also complete and show useful scaling behavior.

| Scale | Cases | NPQR completed | NPQR lower than SABRE basic | Role |
| --- | ---: | ---: | ---: | --- |
| Representative 10/20Q | 10 | 10 | 10 | Main quality evidence. |
| Selected 30/50Q | 4 | 4 | 4 | Bounded scaling evidence. |
| 80/100Q boundary rows | 4 | 0 | 0 | Boundary definition only. |

The project does not claim that NPQR beats SABRE on every circuit. It does not
claim state-of-the-art quantum routing. SABRE lookahead is not the primary
comparison target. The current demonstrated completion boundary is selected
50-qubit structures under the documented test budget.

## Deployment

Build and run the REST API container:

```bash
docker build -f Dockerfile.api -t zju-quantum-compiler-api .
docker run --rm -p 8080:8080 zju-quantum-compiler-api
```

Build and run the MCP container:

```bash
docker build -f Dockerfile.mcp -t zju-quantum-compiler-mcp .
docker run --rm -p 8081:8081 zju-quantum-compiler-mcp
```

Render blueprints are provided in `render.yaml` and `render-mcp.yaml`.

## Verification

Run the public test set before publishing changes:

```bash
python -m pytest -q \
  tests/test_public_api.py \
  tests/test_public_site.py \
  tests/test_public_deploy_config.py \
  tests/test_readme_contract.py \
  tests/test_submission_readiness.py \
  tests/test_algorithm_matrix.py \
  tests/test_demo_scripts.py \
  tests/test_playground_user_guide.py
python scripts/check_submission_readiness.py
git diff --check
```

Generate a local review package:

```bash
python scripts/package_submission.py
```

Generated files are written to `results/submission_package/` and are not
committed.

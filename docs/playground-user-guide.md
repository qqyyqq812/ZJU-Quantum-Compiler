# Quantum Compiler Playground user guide

This guide explains how to review the public browser console for the 量子信息基础大作业.
The page is designed for a quick first pass: open it, run a small example, and
inspect the route trace, QASM output, topology, and NPQR/SABRE comparison.

## Three-minute review

Use this flow for the first browser check.

1. Open the GitHub Pages console or local `docs/index.html`.
2. Select `ghz5`, `qft5`, or `qaoa5`.
3. Click **Run**.
4. Read the NPQR and SABRE columns: status, SWAP count, depth, and `elapsed_ms`.
5. Click **Step** to move through the route trace.
6. Switch between **NPQR QASM** and **SABRE QASM** to inspect the routed output.
7. Check that the topology panel explains which physical edge is used by each
   gate or inserted SWAP.

The 5Q and 10Q examples are the recommended first pass. They load quickly and
make the route trace easy to inspect. The 30Q and 50Q examples are
extension-scale examples; `LineGHZ30` and `Random30-d4` use `grid_5x6`, and
`LineGHZ50` and `RingSparse50` use `grid_5x10`. Use them when a backend service
is deployed and the runtime budget is sufficient.

## What to look at

The browser page has three review surfaces.

| Surface | What it shows | Why it matters |
| --- | --- | --- |
| QASM input | Checked-in examples, custom OpenQASM, generated circuits | Confirms the compiler input is visible. |
| Topology and trace | Physical graph, gate events, SWAP events, mappings | Explains how routing satisfies hardware coupling. |
| Result columns | NPQR, SABRE, SWAP, depth, elapsed time, routed QASM | Shows the fixed comparison used by the report. |

For a useful review, confirm that the page opens, QASM is readable, the topology
matches the selected scale, SWAP replay explains the route, and the NPQR/SABRE
comparison is visible without reading source code.

## Inputs

The left panel controls the circuit.

- **Examples** loads checked-in OpenQASM examples. Use this for repeatable
  review. The 5Q and 10Q examples use IBM Tokyo 20Q. The 30Q and 50Q examples
  use grid topologies.
- **Custom QASM** lets you paste OpenQASM 2 text. The input must start with
  `OPENQASM 2.0;`.
- **Generate** writes a small OpenQASM circuit into the editor. Click **Run** to
  send it to the compiler.

## GitHub Pages, REST API, and MCP

These surfaces serve different users.

| Surface | Audience | Role |
| --- | --- | --- |
| GitHub Pages | People reading and trying the project | Human-facing browser entry. |
| REST API | Browser page, scripts, deployments | Real compiler backend for NPQR and SABRE calls. |
| MCP | Tool clients and automation | Advanced interface; not required for normal browser use. |

The browser can use an HTTPS REST API when one is configured. A local HTTP API
can be used from a local page, but published GitHub Pages may block plain HTTP
requests through browser security rules. Use the API input or `?api=` query
parameter when testing a deployed backend.

## Results

The result columns show the main metrics.

- **Status** reports whether the route finished.
- **SWAP** counts the inserted SWAP gates.
- **Depth** reports routed circuit depth.
- **elapsed_ms** reports backend compute time for that request.
- **Delta values** compare NPQR with SABRE for the selected run.

The **compiled_qasm** panel shows the routed output circuit. Use **NPQR QASM**
and **SABRE QASM** to switch between outputs.

## Topology and route trace

The topology panel shows physical qubits and coupling edges. A two-qubit gate
can run directly only when the mapped physical qubits are adjacent. If they are
not adjacent, the compiler inserts SWAP gates along valid hardware edges.

The route trace is returned in the `route_trace` field.

- A `gate` event means a routed operation can run at the current physical
  locations.
- A `swap` event means the compiler inserted a SWAP before a later gate.
- `logical_qubits` and `physical_qubits` identify the operation.
- `mapping_before` and `mapping_after` show how logical qubits move.

The trace tabs let you inspect NPQR and SABRE routes separately.

## Backend algorithms

NPQR is the project compiler. It parses OpenQASM, builds gate dependencies,
selects initial mappings, scores legal SWAP actions, keeps bounded search
candidates, repairs difficult suffixes, and replays the final trace before
returning routed QASM.

SABRE basic is the fixed Qiskit baseline. The comparison uses the same input circuit, topology, and metric fields for both compilers.

## Troubleshooting

- If the page opens but compile fails, check whether the configured REST API is
  reachable from the browser.
- If GitHub Pages is used with a local HTTP API, browser mixed-content rules may
  block the request.
- If custom QASM fails immediately, check that the first line is
  `OPENQASM 2.0;`.
- If a 30Q or 50Q example takes too long, return to a 5Q or 10Q example for the
  first review and use a deployed backend for extension-scale runs.

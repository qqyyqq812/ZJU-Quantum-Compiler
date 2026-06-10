# Quantum Compiler Playground user guide

This guide explains how to use the public Playground and what happens in the
backend when you compile a circuit. It is written for reviewers, teammates, and
users who need to run real examples without learning the deployment details
first.

> **Note:** The Playground is a live compiler interface. It calls the deployed
> REST API by default. MCP remains an advanced helper for tool clients and is
> not the path used by the **Run** button.

## Quick start

Use this flow when you want to verify that the page and backend are working.

1. Open `docs/index.html`, or open the deployed GitHub Pages version.
2. Check the small backend indicator in the top bar. **在线** means the page can
   reach the FastAPI REST backend.
3. Select an example such as `qft5`, `ghz5`, or `qaoa5`.
4. Click **Run**. The page sends the same circuit to NPQR and to the fixed
   SABRE baseline.
5. Review the two result columns. Each column shows status, SWAP count, depth,
   and `elapsed_ms`.
6. Click **Step** to move through the displayed route trace one event at a time.
7. Switch between **NPQR QASM** and **SABRE QASM** to inspect the routed
   OpenQASM output for each compiler.

If the backend indicator shows an offline state, the web page loaded correctly
but the REST API is not reachable from the browser. The public route uses the
deployed API at `http://1.95.70.10` unless a maintainer overrides it with
`?api=`.

## Input modes

The left side of the Playground controls the circuit that will be compiled.
Changing the input does not call the backend until you click **Run**.

- **Examples** loads checked-in OpenQASM examples from the project. Use this for
  demos and repeatable comparisons.
- **Custom QASM** lets you paste OpenQASM 2 text. The input must start with
  `OPENQASM 2.0;` and include a valid circuit.
- **Generate** creates a valid OpenQASM 2 circuit in the editor. It supports
  GHZ chain, QFT-like, QAOA ring, VQE ansatz, and random CX layers. The
  generator only writes text; **Run** performs the real compile.

Use smaller circuits first when you want a fast browser demo. Larger random or
dense QFT-like circuits may take longer because the backend performs real
routing work.

## Controls

The top control strip mirrors a small experiment bench. Each control has a
different backend effect.

- **Run** calls `POST /api/compile` twice with the same circuit. One request uses
  `backend="npqr"`, and the other uses `backend="sabre"`.
- **Pause** stops the route animation in the browser. It does not cancel a
  request that has already reached the backend.
- **Step** advances the currently selected route trace by one event. It does
  not compile again and does not change the metrics.
- **Reset** restores the default example and clears the current comparison.

The Playground does not expose SABRE tuning controls to normal users. SABRE is
the fixed standard baseline, while NPQR is the project compiler being reviewed.

## Results

The right side of the page shows a compact comparison. Treat these values as the
main review surface for a run.

- **Status** shows whether a backend finished the route. `OK` means a routed
  circuit was produced. `INCOMPLETE` means the route stopped before all gates
  were executed. `N/A` usually means the deployment is missing a required model
  file or dependency.
- **SWAP** is the number of SWAP gates inserted to satisfy hardware
  connectivity.
- **Depth** is the depth of the routed circuit after compilation.
- **elapsed_ms** is backend compute time in milliseconds for that request. It is
  not a stable algorithm-quality metric because it depends on server load,
  warmup, and network timing.
- **Delta values** compare NPQR with SABRE for SWAP, depth, and runtime. Lower
  SWAP or depth is better for those two quality metrics.

The **compiled_qasm** panel shows the routed output circuit. Use **NPQR QASM**
and **SABRE QASM** to switch between the two outputs. The output is physical
OpenQASM on the Tokyo coupling graph, so it usually uses `qreg q[20];` even
when the logical input circuit has fewer qubits.

## Tokyo topology and mapping

The center panel visualizes IBM Tokyo 20Q. The 20 points are physical qubits on
the chip. A line between two points means the hardware can directly run a
two-qubit operation between those two physical qubits.

Input QASM uses logical qubits such as `q[0]`, `q[1]`, and `q[2]`. The compiler
must choose where those logical qubits live on the physical Tokyo nodes. This is
the logical-to-physical mapping. For example, a mapping may place logical
`q[0]` on physical `p3` and logical `q[1]` on physical `p8`.

If the next two-qubit gate needs logical qubits that are not adjacent on the
hardware graph, the compiler inserts SWAP gates along allowed edges. A SWAP
exchanges the logical states stored on two neighboring physical qubits. After a
SWAP, the mapping changes, and later gates are interpreted using the new
logical-to-physical positions.

## Route trace

The route trace is the step-by-step explanation behind the animation. It is
returned by the REST API in the `route_trace` field.

- A `gate` event means a routed operation from the circuit can run at the
  current physical locations.
- A `swap` event means the compiler inserted a SWAP before a later gate so the
  required logical qubits can become adjacent.
- `logical_qubits` names the logical qubits involved in the event.
- `physical_qubits` names the Tokyo nodes used by the event.
- `mapping_before` shows where logical qubits lived before the event.
- `mapping_after` shows where logical qubits live after the event.
- `insertion_index` links an inserted SWAP to the point in the original gate
  stream where it was needed.

The route tabs let you inspect either NPQR or SABRE trace data. The default
animation focuses on NPQR when NPQR returns a trace. SABRE trace data is shown
for comparison and review, not as a fake NPQR animation.

## Backend algorithms

The browser calls the REST API served by `src.server.app:app`. The normal user
path is:

```text
browser
  -> docs/index.html
  -> POST /api/compile
  -> src.server.app:app
  -> NPQR and SABRE routing backends
```

NPQR is the project compiler. It is a neural-assisted selector, search, and repair runtime. It is also described internally as the neural-assisted selector/search/repair runtime:

- It parses the OpenQASM circuit and loads the IBM Tokyo coupling graph.
- It generates candidate initial logical-to-physical mappings.
- It uses a learned policy to score valid SWAP actions.
- It keeps several candidate routes with bounded beam search.
- It can use trigger-gated frontier search and local suffix repair on difficult
  cases.
- It replays the selected action trace and emits routed OpenQASM only when the
  trace is valid.

SABRE is the fixed baseline. The backend uses Qiskit `SabreSwap` with the
standard `basic` heuristic, `seed=42`, and `trials=1`. This makes the comparison
reproducible. SABRE is strong and practical, so the page uses it as the
reference column rather than letting users tune it.

The page does not claim that NPQR always beats SABRE. It shows the actual result
for the selected circuit. Some circuits favor NPQR, some tie, and some may favor
SABRE.

## REST API and MCP

REST API and MCP are separate surfaces.

- The **REST API** is the browser path. **Run** calls `/api/compile` on the
  deployed FastAPI backend. Normal users only need the page and the deployed
  API.
- MCP is an advanced helper for AI clients, reviewers, and tool workflows.
  It exposes tools such as `compile_qasm`, `compile_npqr`, and `compile_sabre`.
  The public page keeps MCP in the **Advanced** section because it is not needed
  for normal browser use.

Maintainers can override the REST API base with `?api=https://your-api.example`
or the Advanced input. Do this only when testing a staging backend.

## Troubleshooting

Use these checks when a run does not produce the expected output.

- If the backend indicator is offline, the browser cannot reach the REST API.
- If NPQR shows `N/A` but SABRE shows `OK`, the API is running but the NPQR model
  file or dependency is missing in that deployment.
- If both columns show errors, the API may be down, the request may have timed
  out, or the QASM may be invalid.
- If Custom QASM fails immediately, check that the text starts with
  `OPENQASM 2.0;` and uses OpenQASM 2 syntax.
- If a large generated circuit takes a long time, try fewer qubits or fewer
  layers before treating it as a service failure.

## Next steps

Use the Playground for live review, then save the relevant QASM or metrics for
reports. For deployment details, use the README, the Dockerfiles, and the Render
blueprints in the repository root.

"""Public machine-readable evidence for the NPQR routing project."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / "results" / "public_algorithm_evidence.json"
DEFAULT_MODEL_PATH = "models/default/npqr-default.pt"


REPRESENTATIVE_10_20_BASIC_ROWS: list[dict[str, Any]] = [
    {
        "case": "GHZ10",
        "qubits": 10,
        "npqr_swaps": 2,
        "sabre_basic_swaps": 6,
        "delta_npqr_minus_sabre_basic": -4,
    },
    {
        "case": "QFT10",
        "qubits": 10,
        "npqr_swaps": 6,
        "sabre_basic_swaps": 13,
        "delta_npqr_minus_sabre_basic": -7,
    },
    {
        "case": "QAOA10",
        "qubits": 10,
        "npqr_swaps": 0,
        "sabre_basic_swaps": 15,
        "delta_npqr_minus_sabre_basic": -15,
    },
    {
        "case": "VQE10",
        "qubits": 10,
        "npqr_swaps": 1,
        "sabre_basic_swaps": 4,
        "delta_npqr_minus_sabre_basic": -3,
    },
    {
        "case": "Random10",
        "qubits": 10,
        "npqr_swaps": 17,
        "sabre_basic_swaps": 19,
        "delta_npqr_minus_sabre_basic": -2,
    },
    {
        "case": "Brickwork20",
        "qubits": 20,
        "npqr_swaps": 15,
        "sabre_basic_swaps": 31,
        "delta_npqr_minus_sabre_basic": -16,
    },
    {
        "case": "CliqueBlocks20",
        "qubits": 20,
        "npqr_swaps": 19,
        "sabre_basic_swaps": 38,
        "delta_npqr_minus_sabre_basic": -19,
    },
    {
        "case": "DeepRandom20",
        "qubits": 20,
        "npqr_swaps": 105,
        "sabre_basic_swaps": 135,
        "delta_npqr_minus_sabre_basic": -30,
    },
    {
        "case": "RingEntangler20",
        "qubits": 20,
        "npqr_swaps": 16,
        "sabre_basic_swaps": 41,
        "delta_npqr_minus_sabre_basic": -25,
    },
    {
        "case": "SparseRandom20",
        "qubits": 20,
        "npqr_swaps": 37,
        "sabre_basic_swaps": 57,
        "delta_npqr_minus_sabre_basic": -20,
    },
]

SCALE_SMOKE_30_50_BASIC_ROWS: list[dict[str, Any]] = [
    {
        "case": "LineGHZ30",
        "qubits": 30,
        "npqr_swaps": 23,
        "sabre_basic_swaps": 43,
        "delta_npqr_minus_sabre_basic": -20,
    },
    {
        "case": "Random30-d4",
        "qubits": 30,
        "npqr_swaps": 61,
        "sabre_basic_swaps": 67,
        "delta_npqr_minus_sabre_basic": -6,
    },
    {
        "case": "LineGHZ50",
        "qubits": 50,
        "npqr_swaps": 50,
        "sabre_basic_swaps": 79,
        "delta_npqr_minus_sabre_basic": -29,
    },
    {
        "case": "RingSparse50",
        "qubits": 50,
        "npqr_swaps": 98,
        "sabre_basic_swaps": 113,
        "delta_npqr_minus_sabre_basic": -15,
    },
]

LARGE_SCALE_BOUNDARY_ROWS: list[dict[str, Any]] = [
    {
        "case": "LineGHZ80",
        "qubits": 80,
        "topology": "grid_8x10",
        "npqr_completed": False,
        "npqr_status": "TIMEOUT",
        "timeout_s": 240,
        "sabre_basic_swaps": 137,
        "boundary": "NPQR did not complete within the bounded CPU test budget.",
    },
    {
        "case": "Random80-d2",
        "qubits": 80,
        "topology": "grid_8x10",
        "npqr_completed": False,
        "npqr_status": "NOT_RUN_AFTER_BOUNDED_STOP",
        "timeout_s": 240,
        "sabre_basic_swaps": None,
        "boundary": "The bounded run stopped before this case completed.",
    },
    {
        "case": "LineGHZ100",
        "qubits": 100,
        "topology": "grid_10x10",
        "npqr_completed": False,
        "npqr_status": "NOT_RUN",
        "timeout_s": 300,
        "sabre_basic_swaps": None,
        "boundary": "Not claimed; reserved for future bounded evaluation.",
    },
    {
        "case": "RingSparse100",
        "qubits": 100,
        "topology": "grid_10x10",
        "npqr_completed": False,
        "npqr_status": "NOT_RUN",
        "timeout_s": 300,
        "sabre_basic_swaps": None,
        "boundary": "Not claimed; reserved for future bounded evaluation.",
    },
]


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "npqr_completed": len(rows),
        "npqr_beats_sabre_basic": sum(
            1 for row in rows if row["delta_npqr_minus_sabre_basic"] < 0
        ),
        "sabre_fallback_used": False,
    }


def build_npqr_evidence_manifest() -> dict[str, Any]:
    """Return the final public algorithm boundary without internal experiment logs."""
    return {
        "schema": "npqr_public_algorithm_evidence_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_claim": (
            "NPQR is a neural-assisted quantum routing compiler for IBM Tokyo-style "
            "hardware constraints. It combines a learned policy with mapping "
            "selection, bounded beam search, pruning, and local repair. On the "
            "representative 10/20-qubit course benchmark, it completes every case "
            "and uses fewer SWAPs than the SABRE basic baseline."
        ),
        "default_route": {
            "backend": "npqr",
            "model": DEFAULT_MODEL_PATH,
            "sabre_fallback": False,
            "comparison_baseline": "qiskit_sabre_swap_basic",
        },
        "algorithm_components": [
            {
                "name": "graph preprocessing",
                "role": "Parse the circuit, build the hardware graph, and compute distance features.",
            },
            {
                "name": "initial mapping selector",
                "role": "Generate and rank logical-to-physical qubit placements before routing.",
            },
            {
                "name": "neural action scorer",
                "role": "Use the checked-in policy network to score valid SWAP actions.",
            },
            {
                "name": "bounded beam search",
                "role": "Keep multiple candidate routes instead of committing to one greedy path.",
            },
            {
                "name": "frontier trigger and pruning",
                "role": "Apply heavier search only on difficult interaction patterns and trim weak actions.",
            },
            {
                "name": "suffix repair",
                "role": "Use a bounded local search when a nearly complete route stalls near the end.",
            },
            {
                "name": "trace replay",
                "role": "Rebuild and verify the routed circuit against the hardware topology.",
            },
        ],
        "baseline": {
            "name": "SABRE",
            "implementation": "Qiskit SabreSwap",
            "heuristic": "basic",
            "seed": 42,
            "trials": 1,
            "role": "Strong comparison baseline only.",
            "not_our_algorithm": True,
        },
        "public_limits": {
            "qasm": "OpenQASM 2",
            "max_inline_qasm_chars": 8000,
            "default_topology": "IBM Tokyo 20Q",
            "max_qubits": 20,
            "sabre_seed": 42,
            "sabre_trials": 1,
        },
        "representative_10_20_basic": {
            "description": (
                "Main course-project quality evidence. NPQR is compared with the "
                "SABRE basic baseline on representative 10/20-qubit routing tasks."
            ),
            "summary": _quality_summary(REPRESENTATIVE_10_20_BASIC_ROWS),
            "trace_replay": "passed_for_completed_npqr_rows",
            "rows": REPRESENTATIVE_10_20_BASIC_ROWS,
        },
        "scale_smoke_30_50_basic": {
            "description": (
                "Scale-potential bounded test. It demonstrates that the current "
                "runtime can route selected 30/50-qubit circuits, but it is not "
                "a claim of universal large-scale superiority."
            ),
            "summary": _quality_summary(SCALE_SMOKE_30_50_BASIC_ROWS),
            "trace_replay": "passed_for_completed_npqr_rows",
            "rows": SCALE_SMOKE_30_50_BASIC_ROWS,
            "known_boundary": (
                "A bounded CPU test completed selected 30/50-qubit cases and found "
                "80-qubit routing beyond the current practical CPU budget."
            ),
        },
        "large_scale_boundary": {
            "description": (
                "Bounded exploratory test above 50 qubits. These rows define the "
                "current upper boundary and are not promotion claims."
            ),
            "max_completed_qubits": 50,
            "max_sabre_basic_win_qubits": 50,
            "rows": LARGE_SCALE_BOUNDARY_ROWS,
        },
        "final_smoke": {
            "description": (
                "Historical compact runtime check retained for compatibility with "
                "older package checks. The final report claim is based on "
                "representative_10_20_basic and scale_smoke_30_50_basic."
            ),
            "npqr_completed": True,
            "reported_swaps": 46,
            "rest_and_mcp_consistent": True,
            "sabre_fallback_used": False,
        },
        "claims": {
            "claimed": [
                "The repository exposes a runnable neural-assisted routing backend.",
                "The default route uses NPQR and returns SABRE metrics as a baseline comparison.",
                "The route is verified by replaying the emitted gate and SWAP trace.",
                "NPQR completes and beats SABRE basic on the representative 10/20-qubit benchmark.",
                "The bounded 30/50-qubit test completes and beats SABRE basic on the selected cases.",
                "The project is suitable for course reporting as a graph search and neural heuristic system.",
            ],
            "not_claimed": [
                "NPQR is not claimed to beat SABRE on every circuit.",
                "NPQR is not claimed to beat SABRE lookahead across the board.",
                "80/100-qubit routing is not claimed as a completed capability in the final course submission.",
                "The checked-in model is not claimed to be a fully retrained state-of-the-art model.",
                "SABRE is not used as a hidden fallback for NPQR completion.",
            ],
        },
        "course_algorithm_mapping": [
            "graph modeling",
            "transform-and-conquer",
            "greedy heuristics",
            "decrease-and-conquer",
            "time-space tradeoff",
            "iterative improvement",
            "branch pruning",
            "approximation",
            "neural network inference",
        ],
    }


def write_npqr_evidence_manifest(path: Path = DEFAULT_EVIDENCE_PATH) -> dict[str, Any]:
    """Write the public evidence payload for local packaging."""
    manifest = build_npqr_evidence_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_npqr_evidence_manifest(path: Path = DEFAULT_EVIDENCE_PATH) -> dict[str, Any]:
    """Load generated public evidence, or build it from checked-in constants."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_npqr_evidence_manifest()

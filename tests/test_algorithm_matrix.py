"""Contract tests for the reproducible algorithm experiment matrix."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.experiment_algorithm_matrix import run_matrix


def test_algorithm_matrix_quick_rows_match_public_static_defaults():
    rows = run_matrix(quick=True)

    assert rows
    assert {row.algorithm for row in rows} == {"basic", "lookahead", "decay"}
    assert {row.circuit for row in rows} == {"ghz5", "qft5", "qft10"}
    assert all(row.completed for row in rows)
    assert all(row.runtime_ms >= 0 for row in rows)

    qft10_lookahead = next(
        row for row in rows if row.circuit == "qft10" and row.algorithm == "lookahead"
    )
    assert qft10_lookahead.swap == 29
    assert qft10_lookahead.depth == 156


def test_algorithm_matrix_cli_json_output_is_machine_readable():
    output = subprocess.check_output(
        [
            sys.executable,
            "scripts/experiment_algorithm_matrix.py",
            "--quick",
            "--json",
        ],
        text=True,
    )
    data = json.loads(output)

    assert isinstance(data, list)
    assert data
    for row in data:
        assert set(row) == {
            "circuit",
            "algorithm",
            "swap",
            "depth",
            "completed",
            "runtime_ms",
        }


def test_algorithm_matrix_cli_csv_output(tmp_path):
    output_path = tmp_path / "algorithm_matrix.csv"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/experiment_algorithm_matrix.py",
            "--quick",
            "--csv",
            str(output_path),
        ]
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0].keys() == {
        "circuit",
        "algorithm",
        "swap",
        "depth",
        "completed",
        "runtime_ms",
    }

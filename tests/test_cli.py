"""CLI smoke tests for teacher-facing entry points."""
from __future__ import annotations

from qiskit import QuantumCircuit, qasm2

from src.cli import main


def test_qcompiler_info_reports_current_status(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Current status" in out
    assert "checkpoint_ep25333.pt" in out


def test_qcompiler_eval_supports_ghz_without_ai(capsys):
    assert main(["eval", "--circuits", "ghz_3", "--topology", "tokyo", "--no-ai"]) == 0
    out = capsys.readouterr().out
    assert "ghz_3" in out
    assert "SABRE" in out


def test_qcompiler_compile_sabre_writes_output(tmp_path, capsys):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qasm_path = tmp_path / "demo.qasm"
    out_path = tmp_path / "compiled.qasm"
    qasm2.dump(qc, qasm_path)

    assert main([
        "compile",
        str(qasm_path),
        "--topology",
        "tokyo",
        "--backend",
        "sabre",
        "--output",
        str(out_path),
    ]) == 0
    out = capsys.readouterr().out
    assert "Compiled:" in out
    assert out_path.exists()

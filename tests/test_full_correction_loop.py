from __future__ import annotations

import json
from pathlib import Path

from run_collective_lab_validation_demo import run_demo


def test_full_demo_correction_loop_succeeds(tmp_path: Path) -> None:
    messages: list[str] = []
    summary = run_demo(
        mode="demo",
        no_pause=True,
        artifacts_dir=tmp_path,
        output=messages.append,
    )
    terminal = "\n".join(messages)
    assert "REJECTED: unit conversion mismatch" in terminal
    assert "BLOCKED: JAX execution denied" in terminal
    assert "JAX wurde nicht gestartet." in terminal
    assert "VERIFIED: parameters accepted" in terminal
    assert "APPROVED: JAX execution allowed" in terminal
    assert summary.verified_payload.matrix == [[1.4, 0.0], [0.0, 1.5]]
    assert summary.simulation.completed

    required = {
        "source_record.json",
        "extracted_payload.json",
        "rejected_audit_report.json",
        "blocked_gate_decision.json",
        "corrected_payload.json",
        "verified_audit_report.json",
        "approved_gate_decision.json",
        "verified_conductivity_matrix.json",
        "simulation_result.json",
        "jax_heat_result.png",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    verified = json.loads(
        (tmp_path / "verified_conductivity_matrix.json").read_text(encoding="utf-8")
    )
    assert verified["validation_status"] == "VERIFIED"

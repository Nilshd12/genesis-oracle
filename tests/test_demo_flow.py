from __future__ import annotations

import json

import pytest

from run_collective_lab_demo import run_demo


def test_complete_demo_flow_succeeds(tmp_path) -> None:
    messages: list[str] = []
    summary = run_demo(
        mode="demo",
        no_pause=True,
        artifacts_dir=tmp_path,
        output=messages.append,
    )

    terminal = "\n".join(messages)
    assert summary.success
    assert summary.final_parameter.value == pytest.approx(1.5)
    assert summary.simulation.completed
    assert "Reproduzierbarer Demofehler wird injiziert." in terminal
    assert "REJECTED: unit conversion mismatch" in terminal
    assert "BLOCKED: JAX execution denied" in terminal
    assert "VERIFIED: parameters accepted" in terminal
    assert "APPROVED: JAX execution allowed" in terminal

    expected_artifacts = {
        "source_record.json",
        "extracted_parameters.json",
        "rejected_audit_report.json",
        "corrected_parameters.json",
        "verified_audit_report.json",
        "gate_decision.json",
        "simulation_result.json",
    }
    assert expected_artifacts.issubset({path.name for path in tmp_path.iterdir()})

    rejected = json.loads((tmp_path / "rejected_audit_report.json").read_text())
    corrected = json.loads((tmp_path / "corrected_parameters.json").read_text())
    assert rejected["status"] == "REJECTED"
    assert corrected["value"] == pytest.approx(1.5)
    assert corrected["correction_history"]

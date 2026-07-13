from __future__ import annotations

import pytest

from collective_lab_demo.agents.auditor_agent import AuditorAgent
from collective_lab_demo.models.parameter_models import (
    AuditReport,
    GateDecision,
    GateStatus,
    ParameterRecord,
    WorkflowStatus,
)
from collective_lab_demo.simulation.jax_heat_simulation import (
    run_heat_simulation,
)
from collective_lab_demo.validation.gate import evaluate_gate


def make_parameter(
    *,
    validation_status: WorkflowStatus = WorkflowStatus.VERIFIED,
    source_title: str = "Verified source",
) -> ParameterRecord:
    return ParameterRecord(
        parameter_name="thermal_conductivity",
        value=1.5,
        unit="W/(m*K)",
        source_value=0.015,
        source_unit="W/(cm*K)",
        source_title=source_title,
        source_url="https://doi.org/10.1364/AO.33.001000",
        source_excerpt="Measured value: 0.015 W/(cm K).",
        validation_status=validation_status,
    )


def report(status: WorkflowStatus) -> AuditReport:
    return AuditReport(
        status=status,
        passed_checks=[],
        failed_checks=[] if status is WorkflowStatus.VERIFIED else ["rejected"],
        audit_summary=status.value,
    )


def test_missing_source_is_rejected() -> None:
    parameter = make_parameter(source_title="")
    audit = AuditorAgent().audit(parameter)
    assert audit.status is WorkflowStatus.REJECTED
    assert "missing source evidence" in audit.failed_checks


def test_gate_blocks_rejected_dataset() -> None:
    decision = evaluate_gate(make_parameter(), report(WorkflowStatus.REJECTED))
    assert decision.status is GateStatus.BLOCKED
    assert not decision.execution_allowed


def test_gate_accepts_exclusively_verified_dataset() -> None:
    pending = make_parameter(validation_status=WorkflowStatus.PENDING)
    assert not evaluate_gate(pending, report(WorkflowStatus.VERIFIED)).execution_allowed

    verified = make_parameter()
    decision = evaluate_gate(verified, report(WorkflowStatus.VERIFIED))
    assert decision.status is GateStatus.APPROVED
    assert decision.execution_allowed


def test_simulation_refuses_missing_gate_approval() -> None:
    denied = GateDecision(
        status=GateStatus.BLOCKED,
        execution_allowed=False,
        reasons=["not approved"],
    )
    with pytest.raises(PermissionError, match="simulation denied"):
        run_heat_simulation(make_parameter(), denied)


def test_simulation_runs_with_matching_verified_approval() -> None:
    parameter = make_parameter()
    decision = evaluate_gate(parameter, report(WorkflowStatus.VERIFIED))
    result = run_heat_simulation(parameter, decision, steps=3, grid_points=9)
    assert result.completed
    assert result.thermal_conductivity == pytest.approx(1.5)
    assert result.steps == 3

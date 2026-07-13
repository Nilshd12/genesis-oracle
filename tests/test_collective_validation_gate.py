from __future__ import annotations

from collective_lab_validation.agents.auditor_agent import AuditorAgent
from collective_lab_validation.models.gate_decision import GateStatus
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)
from collective_lab_validation.validation.deterministic_gate import evaluate_gate


def test_gate_blocks_rejected_payload(
    wrong_tensor_payload: ParameterPayload,
) -> None:
    audit = AuditorAgent().audit(wrong_tensor_payload)
    rejected = wrong_tensor_payload.model_copy(
        update={"validation_status": WorkflowStatus.REJECTED}
    )
    decision = evaluate_gate(rejected, audit)
    assert decision.status is GateStatus.BLOCKED
    assert not decision.execution_allowed


def test_gate_accepts_only_verified_payload(
    corrected_tensor_payload: ParameterPayload,
    verified_tensor_payload: ParameterPayload,
) -> None:
    audit = AuditorAgent().audit(corrected_tensor_payload)
    assert not evaluate_gate(corrected_tensor_payload, audit).execution_allowed
    decision = evaluate_gate(verified_tensor_payload, audit)
    assert decision.status is GateStatus.APPROVED
    assert decision.execution_allowed

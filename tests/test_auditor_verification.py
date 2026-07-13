from __future__ import annotations

from collective_lab_validation.agents.auditor_agent import AuditorAgent
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)


def test_correction_loop_produces_expected_tensor(
    corrected_tensor_payload: ParameterPayload,
) -> None:
    assert corrected_tensor_payload.matrix == [[1.4, 0.0], [0.0, 1.5]]
    assert corrected_tensor_payload.correction_history
    assert corrected_tensor_payload.extraction_status is WorkflowStatus.CORRECTED


def test_second_audit_verifies_corrected_tensor(
    corrected_tensor_payload: ParameterPayload,
) -> None:
    report = AuditorAgent().audit(corrected_tensor_payload)
    assert report.status is WorkflowStatus.VERIFIED
    assert report.audit_summary == "VERIFIED: parameters accepted"
    assert all(report.deterministic_checks.values())

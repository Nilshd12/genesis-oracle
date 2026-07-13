from __future__ import annotations

from collective_lab_validation.agents.auditor_agent import AuditorAgent
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)


def test_first_audit_rejects_mislabeled_tensor(
    wrong_tensor_payload: ParameterPayload,
) -> None:
    report = AuditorAgent().audit(wrong_tensor_payload)
    assert report.status is WorkflowStatus.REJECTED
    assert report.audit_summary == "REJECTED: unit conversion mismatch"
    assert report.expected_matrix == [[1.4, 0.0], [0.0, 1.5]]
    assert "unit conversion mismatch" in report.failed_checks


def test_missing_source_is_rejected(
    wrong_tensor_payload: ParameterPayload,
) -> None:
    without_source = wrong_tensor_payload.model_copy(
        update={"source_title": "", "source_doi": "", "source_url": ""}
    )
    report = AuditorAgent().audit(without_source)
    assert report.status is WorkflowStatus.REJECTED
    assert not report.deterministic_checks["source_valid"]

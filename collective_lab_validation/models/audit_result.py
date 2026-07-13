"""Structured output of the skeptical Auditor-Agent."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from collective_lab_validation.models.parameter_payload import (
    Matrix,
    StrictModel,
    WorkflowStatus,
    utc_now,
)


class AuditResult(StrictModel):
    status: WorkflowStatus
    passed_checks: list[str]
    failed_checks: list[str]
    expected_matrix: Matrix | None = None
    received_matrix: Matrix | None = None
    expected_unit: str | None = None
    received_unit: str | None = None
    correction_instruction: str | None = None
    audit_summary: str
    deterministic_checks: dict[str, bool]
    timestamp: datetime = Field(default_factory=utc_now)

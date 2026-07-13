"""Non-language-model gate that repeats every mandatory objective check."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from collective_lab_validation.models.audit_result import AuditResult
from collective_lab_validation.models.gate_decision import GateDecision, GateStatus
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    WorkflowStatus,
)
from collective_lab_validation.validation.matrix_validator import (
    validate_conductivity_matrix,
)
from collective_lab_validation.validation.schema_validator import (
    validate_payload_schema,
)
from collective_lab_validation.validation.source_validator import validate_source
from collective_lab_validation.validation.unit_validator import (
    validate_matrix_conversion,
)


def payload_fingerprint(payload: ParameterPayload) -> str:
    serialized = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def evaluate_gate(
    untrusted: ParameterPayload | Mapping[str, Any],
    audit: AuditResult,
) -> GateDecision:
    reasons: list[str] = []
    schema = validate_payload_schema(untrusted)
    if not schema.valid or schema.payload is None:
        return GateDecision(
            status=GateStatus.BLOCKED,
            execution_allowed=False,
            reasons=["invalid JSON schema", *schema.errors],
        )
    payload = schema.payload
    source = validate_source(payload)
    matrix = validate_conductivity_matrix(payload.matrix)
    conversion = validate_matrix_conversion(
        payload.source_values,
        payload.source_unit,
        payload.matrix,
        payload.unit,
    )
    if audit.status is not WorkflowStatus.VERIFIED:
        reasons.append("Auditor status is not VERIFIED")
    if payload.validation_status is not WorkflowStatus.VERIFIED:
        reasons.append("payload validation_status is not VERIFIED")
    if not source.valid:
        reasons.append("source evidence is invalid")
    if not matrix.valid:
        reasons.append("conductivity matrix is invalid")
    if not conversion.valid:
        reasons.append("unit conversion mismatch")
    if (
        payload.extraction_status is WorkflowStatus.CORRECTED
        and not payload.correction_history
    ):
        reasons.append("correction history is required")
    if not audit.deterministic_checks or not all(
        audit.deterministic_checks.values()
    ):
        reasons.append("not all mandatory deterministic audit checks passed")

    if reasons:
        return GateDecision(
            status=GateStatus.BLOCKED,
            execution_allowed=False,
            reasons=reasons,
        )
    return GateDecision(
        status=GateStatus.APPROVED,
        execution_allowed=True,
        reasons=["all mandatory deterministic checks passed"],
        payload_fingerprint=payload_fingerprint(payload),
    )

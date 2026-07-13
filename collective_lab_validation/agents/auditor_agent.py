"""Skeptical Auditor-Agent backed by objective Python validators."""

from __future__ import annotations

from typing import Any, Mapping

from collective_lab_validation.models.audit_result import AuditResult
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
    TARGET_UNIT,
    validate_matrix_conversion,
)


CORRECTION_INSTRUCTION = (
    "Convert every conductivity value from W/(cm*K) to W/(m*K) by "
    "multiplying by 100. Preserve the source reference and document the "
    "correction."
)


class AuditorAgent:
    name = "Auditor-Agent"

    def audit(
        self,
        untrusted: ParameterPayload | Mapping[str, Any],
    ) -> AuditResult:
        checks: dict[str, bool] = {"schema_valid": False}
        schema = validate_payload_schema(untrusted)
        if not schema.valid or schema.payload is None:
            return AuditResult(
                status=WorkflowStatus.REJECTED,
                passed_checks=[],
                failed_checks=["invalid JSON schema", *schema.errors],
                correction_instruction="Submit a complete ParameterPayload JSON object.",
                audit_summary="REJECTED: invalid JSON schema",
                deterministic_checks=checks,
            )
        payload = schema.payload
        checks["schema_valid"] = True

        source = validate_source(payload)
        checks.update(source.checks)
        checks["source_valid"] = source.valid
        matrix = validate_conductivity_matrix(payload.matrix)
        checks.update(matrix.checks)
        checks["matrix_valid"] = matrix.valid
        conversion = validate_matrix_conversion(
            payload.source_values,
            payload.source_unit,
            payload.matrix,
            payload.unit,
        )
        checks["unit_conversion_correct"] = conversion.valid
        checks["physical_plausibility"] = matrix.valid and all(
            0.001 <= float(payload.matrix[index][index]) <= 1000.0
            for index in range(2)
        )
        checks["correction_history_valid"] = (
            payload.extraction_status is not WorkflowStatus.CORRECTED
            or bool(payload.correction_history)
        )
        checks["no_unreferenced_values"] = conversion.valid

        failed: list[str] = []
        if not source.valid:
            failed.extend(source.errors)
        if not matrix.valid:
            failed.extend(matrix.errors)
        if not conversion.valid:
            failed.append("unit conversion mismatch")
            failed.extend(conversion.errors)
        if not checks["physical_plausibility"]:
            failed.append("matrix values outside allowed physical range")
        if not checks["correction_history_valid"]:
            failed.append("corrected payload has no correction history")

        passed = [name for name, passed_check in checks.items() if passed_check]
        if failed:
            primary = (
                "unit conversion mismatch"
                if "unit conversion mismatch" in failed
                else failed[0]
            )
            return AuditResult(
                status=WorkflowStatus.REJECTED,
                passed_checks=passed,
                failed_checks=list(dict.fromkeys(failed)),
                expected_matrix=conversion.expected_matrix,
                received_matrix=payload.matrix,
                expected_unit=TARGET_UNIT,
                received_unit=payload.unit,
                correction_instruction=CORRECTION_INSTRUCTION,
                audit_summary=f"REJECTED: {primary}",
                deterministic_checks=checks,
            )
        return AuditResult(
            status=WorkflowStatus.VERIFIED,
            passed_checks=passed,
            failed_checks=[],
            expected_matrix=conversion.expected_matrix,
            received_matrix=payload.matrix,
            expected_unit=TARGET_UNIT,
            received_unit=payload.unit,
            audit_summary="VERIFIED: parameters accepted",
            deterministic_checks=checks,
        )

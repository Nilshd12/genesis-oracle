"""Independent, skeptical audit backed by deterministic validators."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from collective_lab_demo.models.parameter_models import (
    AuditReport,
    ParameterRecord,
    WorkflowStatus,
)
from collective_lab_demo.validation.gate import (
    MAX_CONDUCTIVITY_SI,
    MIN_CONDUCTIVITY_SI,
)
from collective_lab_demo.validation.schema_validator import (
    validate_parameter_schema,
)
from collective_lab_demo.validation.unit_validator import (
    CANONICAL_METRE_UNIT,
    convert_value,
    is_supported_unit,
    validate_conversion,
)


class AuditorAgent:
    """Audit untrusted Scholar-Prime payloads and never self-approve."""

    name = "Auditor-Agent"

    def audit(
        self,
        payload: ParameterRecord | Mapping[str, Any],
    ) -> AuditReport:
        """Return a structured report containing every objective failure."""

        passed: list[str] = []
        failed: list[str] = []
        schema = validate_parameter_schema(payload)
        if not schema.valid or schema.record is None:
            failed.append("invalid JSON schema")
            failed.extend(schema.errors)
            return AuditReport(
                status=WorkflowStatus.REJECTED,
                passed_checks=passed,
                failed_checks=failed,
                correction_instruction="Provide a payload matching ParameterRecord.",
                audit_summary="REJECTED: invalid JSON schema",
            )

        record = schema.record
        passed.append("JSON schema and data types are valid")

        source_values = (
            record.source_title,
            record.source_url,
            record.source_excerpt,
        )
        if all(value.strip() for value in source_values):
            passed.append("source evidence is present")
        else:
            failed.append("missing source evidence")

        if isfinite(record.source_value):
            passed.append("source value is present")
        else:
            failed.append("missing or non-finite source value")

        units_valid = is_supported_unit(record.source_unit) and is_supported_unit(
            record.unit
        )
        if units_valid:
            passed.append("source and target units are recognized")
        else:
            failed.append("unsupported conductivity unit")

        if record.source_value > 0 and record.value > 0:
            passed.append("source and payload values are positive")
        else:
            failed.append("conductivity must be positive")

        expected_value: float | None = None
        expected_unit: str | None = None
        if units_valid:
            conversion = validate_conversion(
                record.source_value,
                record.source_unit,
                record.value,
                record.unit,
            )
            expected_value = conversion.expected_value
            expected_unit = conversion.expected_unit
            if conversion.valid:
                passed.append("source-to-payload conversion is correct")
            else:
                failed.append("unit conversion mismatch")

            value_si = convert_value(
                record.value,
                record.unit,
                CANONICAL_METRE_UNIT,
            )
            if MIN_CONDUCTIVITY_SI <= value_si <= MAX_CONDUCTIVITY_SI:
                passed.append("conductivity is physically plausible for the demo")
            else:
                failed.append("conductivity is outside the allowed physical range")

        if record.extraction_status is WorkflowStatus.CORRECTED:
            if record.correction_history:
                passed.append("correction history is present")
            else:
                failed.append("corrected payload has no correction history")
        else:
            passed.append("correction history is not yet required")

        if failed:
            correction_instruction = (
                f"Set value to {expected_value:g} {expected_unit} based on "
                f"{record.source_value:g} {record.source_unit}."
                if expected_value is not None and expected_unit is not None
                else "Correct all failed checks and resubmit the payload."
            )
            primary_failure = (
                "unit conversion mismatch"
                if "unit conversion mismatch" in failed
                else failed[0]
            )
            return AuditReport(
                status=WorkflowStatus.REJECTED,
                passed_checks=passed,
                failed_checks=failed,
                expected_value=expected_value,
                received_value=record.value,
                expected_unit=expected_unit,
                received_unit=record.unit,
                correction_instruction=correction_instruction,
                audit_summary=f"REJECTED: {primary_failure}",
            )

        return AuditReport(
            status=WorkflowStatus.VERIFIED,
            passed_checks=passed,
            failed_checks=[],
            expected_value=expected_value,
            received_value=record.value,
            expected_unit=expected_unit,
            received_unit=record.unit,
            audit_summary="VERIFIED: parameters accepted",
        )

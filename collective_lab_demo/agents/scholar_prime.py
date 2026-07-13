"""Scholar-Prime extraction and audit-directed correction behavior."""

from __future__ import annotations

from collective_lab_demo.models.parameter_models import (
    AuditReport,
    CorrectionRecord,
    ParameterRecord,
    SourceRecord,
    WorkflowStatus,
)
from collective_lab_demo.validation.unit_validator import (
    CANONICAL_METRE_UNIT,
    convert_value,
)


class ScholarPrime:
    """Produce source-grounded payloads without requiring an LLM service."""

    name = "Scholar-Prime"

    def extract(
        self,
        source: SourceRecord,
        *,
        inject_demo_error: bool = False,
    ) -> ParameterRecord:
        """Extract a conductivity and normalize it to W/(m*K)."""

        converted_value = convert_value(
            source.original_value,
            source.original_unit,
            CANONICAL_METRE_UNIT,
        )
        if inject_demo_error:
            # This is deliberate and advertised by the orchestrator. It is
            # never presented as a spontaneous model hallucination.
            converted_value = source.original_value

        return ParameterRecord(
            parameter_name=source.parameter_name,
            value=converted_value,
            unit=CANONICAL_METRE_UNIT,
            source_value=source.original_value,
            source_unit=source.original_unit,
            source_title=source.title,
            source_url=source.url,
            source_excerpt=source.excerpt,
        )

    def correct(
        self,
        rejected: ParameterRecord,
        audit_report: AuditReport,
    ) -> ParameterRecord:
        """Apply only a concrete, deterministic correction from an audit."""

        if audit_report.status is not WorkflowStatus.REJECTED:
            raise ValueError("Correction requires a REJECTED audit report.")
        if (
            audit_report.expected_value is None
            or audit_report.expected_unit is None
            or not audit_report.correction_instruction
        ):
            raise ValueError("The audit report has no actionable correction.")

        correction = CorrectionRecord(
            previous_value=rejected.value,
            previous_unit=rejected.unit,
            corrected_value=audit_report.expected_value,
            corrected_unit=audit_report.expected_unit,
            reason=audit_report.correction_instruction,
        )
        return rejected.model_copy(
            update={
                "value": audit_report.expected_value,
                "unit": audit_report.expected_unit,
                "extraction_status": WorkflowStatus.CORRECTED,
                "validation_status": WorkflowStatus.PENDING,
                "correction_history": [*rejected.correction_history, correction],
            }
        )

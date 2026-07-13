"""Scholar-Prime extraction and audit-directed tensor correction."""

from __future__ import annotations

from collective_lab_validation.models.audit_result import AuditResult
from collective_lab_validation.models.parameter_payload import (
    CorrectionEntry,
    ParameterPayload,
    SourceRecord,
    WorkflowStatus,
)
from collective_lab_validation.validation.unit_validator import (
    TARGET_UNIT,
    convert_matrix,
    source_matrix,
)


class ScholarPrime:
    name = "Scholar-Prime"

    def extract(
        self,
        source: SourceRecord,
        *,
        inject_reproducible_error: bool,
    ) -> ParameterPayload:
        original = source_matrix(source.source_values)
        matrix = (
            original
            if inject_reproducible_error
            else convert_matrix(original, source.source_unit, TARGET_UNIT)
        )
        return ParameterPayload(
            parameter_name="AgGaS2_thermal_conductivity_tensor",
            matrix=matrix,
            unit=TARGET_UNIT,
            source_values=source.source_values,
            source_unit=source.source_unit,
            source_title=source.title,
            source_doi=source.doi,
            source_url=source.source_url,
            source_excerpt=source.source_excerpt,
        )

    def correct(
        self,
        rejected_payload: ParameterPayload,
        audit: AuditResult,
    ) -> ParameterPayload:
        if audit.status is not WorkflowStatus.REJECTED:
            raise ValueError("Scholar-Prime correction requires REJECTED audit feedback.")
        if (
            audit.expected_matrix is None
            or audit.expected_unit is None
            or not audit.correction_instruction
        ):
            raise ValueError("Audit feedback does not contain a deterministic correction.")
        correction = CorrectionEntry(
            previous_matrix=rejected_payload.matrix,
            previous_unit=rejected_payload.unit,
            corrected_matrix=audit.expected_matrix,
            corrected_unit=audit.expected_unit,
            instruction=audit.correction_instruction,
        )
        return rejected_payload.model_copy(
            update={
                "matrix": audit.expected_matrix,
                "unit": audit.expected_unit,
                "extraction_status": WorkflowStatus.CORRECTED,
                "validation_status": WorkflowStatus.PENDING,
                "correction_history": [
                    *rejected_payload.correction_history,
                    correction,
                ],
            }
        )

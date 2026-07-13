"""Hard execution gate between audit/correction and JAX."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from collective_lab_demo.models.parameter_models import (
    AuditReport,
    GateDecision,
    GateStatus,
    ParameterRecord,
    WorkflowStatus,
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


MIN_CONDUCTIVITY_SI = 0.001
MAX_CONDUCTIVITY_SI = 1000.0


def parameter_fingerprint(record: ParameterRecord) -> str:
    """Hash the complete validated payload to bind an approval to it."""

    serialized = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def evaluate_gate(
    payload: ParameterRecord | Mapping[str, Any],
    audit_report: AuditReport,
) -> GateDecision:
    """Approve JAX only if every objective condition passes."""

    reasons: list[str] = []
    schema = validate_parameter_schema(payload)
    if not schema.valid or schema.record is None:
        reasons.append("invalid parameter schema")
        reasons.extend(schema.errors)
        return GateDecision(
            status=GateStatus.BLOCKED,
            execution_allowed=False,
            reasons=reasons,
        )

    record = schema.record
    if audit_report.status is not WorkflowStatus.VERIFIED:
        reasons.append("audit status is not VERIFIED")
    if record.validation_status is not WorkflowStatus.VERIFIED:
        reasons.append("parameter validation status is not VERIFIED")
    if not is_supported_unit(record.source_unit) or not is_supported_unit(
        record.unit
    ):
        reasons.append("unsupported conductivity unit")
    else:
        conversion = validate_conversion(
            record.source_value,
            record.source_unit,
            record.value,
            record.unit,
        )
        if not conversion.valid:
            reasons.append("unit conversion mismatch")

        value_si = convert_value(record.value, record.unit, CANONICAL_METRE_UNIT)
        if not MIN_CONDUCTIVITY_SI <= value_si <= MAX_CONDUCTIVITY_SI:
            reasons.append(
                "conductivity outside allowed range "
                f"[{MIN_CONDUCTIVITY_SI}, {MAX_CONDUCTIVITY_SI}] W/(m*K)"
            )

    if not all(
        value.strip()
        for value in (
            record.source_title,
            record.source_url,
            record.source_excerpt,
        )
    ):
        reasons.append("source evidence is missing")

    if reasons:
        return GateDecision(
            status=GateStatus.BLOCKED,
            execution_allowed=False,
            reasons=reasons,
        )

    return GateDecision(
        status=GateStatus.APPROVED,
        execution_allowed=True,
        reasons=["all deterministic validation conditions passed"],
        parameter_fingerprint=parameter_fingerprint(record),
    )

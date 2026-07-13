"""Deterministic source-evidence checks."""

from __future__ import annotations

from dataclasses import dataclass

from collective_lab_validation.models.parameter_payload import ParameterPayload


@dataclass(frozen=True, slots=True)
class SourceValidationResult:
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]


def validate_source(payload: ParameterPayload) -> SourceValidationResult:
    checks = {
        "source_title_present": bool(payload.source_title.strip()),
        "source_identifier_present": bool(
            payload.source_doi.strip() or payload.source_url.strip()
        ),
        "source_excerpt_present": bool(payload.source_excerpt.strip()),
        "source_values_complete": (
            "parallel" in payload.source_values
            and "perpendicular" in payload.source_values
        ),
        "source_unit_present": bool(payload.source_unit.strip()),
    }
    errors = tuple(name.replace("_", " ") for name, passed in checks.items() if not passed)
    return SourceValidationResult(all(checks.values()), checks, errors)

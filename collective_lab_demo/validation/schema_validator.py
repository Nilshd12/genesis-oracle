"""Pydantic-backed schema validation kept independent of agent text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from collective_lab_demo.models.parameter_models import ParameterRecord


@dataclass(frozen=True, slots=True)
class SchemaValidationResult:
    """Result of validating an untrusted parameter payload."""

    valid: bool
    errors: tuple[str, ...]
    record: ParameterRecord | None = None


def validate_parameter_schema(
    payload: ParameterRecord | Mapping[str, Any],
) -> SchemaValidationResult:
    """Validate types, required fields, enums, and extra fields."""

    try:
        record = (
            payload
            if isinstance(payload, ParameterRecord)
            else ParameterRecord.model_validate(payload)
        )
    except (ValidationError, TypeError) as error:
        if isinstance(error, ValidationError):
            messages = tuple(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
        else:
            messages = (str(error),)
        return SchemaValidationResult(valid=False, errors=messages)

    return SchemaValidationResult(valid=True, errors=(), record=record)

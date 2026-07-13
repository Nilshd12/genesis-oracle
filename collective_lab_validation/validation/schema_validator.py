"""Validate untrusted JSON against the strict ParameterPayload schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from collective_lab_validation.models.parameter_payload import ParameterPayload


@dataclass(frozen=True, slots=True)
class SchemaValidationResult:
    valid: bool
    errors: tuple[str, ...]
    payload: ParameterPayload | None = None


def validate_payload_schema(
    payload: ParameterPayload | Mapping[str, Any],
) -> SchemaValidationResult:
    try:
        validated = (
            payload
            if isinstance(payload, ParameterPayload)
            else ParameterPayload.model_validate(payload)
        )
    except (ValidationError, TypeError) as error:
        if isinstance(error, ValidationError):
            errors = tuple(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
        else:
            errors = (str(error),)
        return SchemaValidationResult(False, errors)
    return SchemaValidationResult(True, (), validated)

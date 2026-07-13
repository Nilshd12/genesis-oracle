from __future__ import annotations

from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.validation.schema_validator import (
    validate_payload_schema,
)


def test_incomplete_json_schema_is_rejected(
    wrong_tensor_payload: ParameterPayload,
) -> None:
    data = wrong_tensor_payload.model_dump(mode="json")
    del data["source_doi"]
    result = validate_payload_schema(data)
    assert not result.valid
    assert any("source_doi" in error for error in result.errors)


def test_missing_target_unit_is_rejected(
    wrong_tensor_payload: ParameterPayload,
) -> None:
    data = wrong_tensor_payload.model_dump(mode="json")
    del data["unit"]
    assert not validate_payload_schema(data).valid

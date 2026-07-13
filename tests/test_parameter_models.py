from __future__ import annotations

from collective_lab_demo.models.parameter_models import ParameterRecord
from collective_lab_demo.validation.schema_validator import (
    validate_parameter_schema,
)


def valid_payload() -> dict[str, object]:
    return {
        "parameter_name": "thermal_conductivity",
        "value": 1.5,
        "unit": "W/(m*K)",
        "source_value": 0.015,
        "source_unit": "W/(cm*K)",
        "source_title": "A paper",
        "source_url": "https://example.test/paper",
        "source_excerpt": "0.015 W/(cm K)",
    }


def test_parameter_model_round_trips_as_json() -> None:
    original = ParameterRecord.model_validate(valid_payload())
    restored = ParameterRecord.model_validate_json(original.model_dump_json())
    assert restored == original


def test_invalid_json_schema_is_rejected() -> None:
    payload = valid_payload()
    del payload["source_unit"]
    payload["unexpected"] = True
    result = validate_parameter_schema(payload)
    assert not result.valid
    assert result.record is None
    assert any("source_unit" in error for error in result.errors)

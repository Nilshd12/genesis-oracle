from __future__ import annotations

import pytest

from collective_lab_demo.agents.auditor_agent import AuditorAgent
from collective_lab_demo.models.parameter_models import (
    ParameterRecord,
    WorkflowStatus,
)
from collective_lab_demo.validation.unit_validator import (
    convert_value,
    validate_conversion,
)


def make_parameter(value: float) -> ParameterRecord:
    return ParameterRecord(
        parameter_name="thermal_conductivity",
        value=value,
        unit="W/(m*K)",
        source_value=0.015,
        source_unit="W/(cm*K)",
        source_title="Verified source",
        source_url="https://doi.org/10.1364/AO.33.001000",
        source_excerpt="Measured value: 0.015 W/(cm K).",
    )


def test_converts_centimetres_to_metres() -> None:
    assert convert_value(0.015, "W/(cm*K)", "W/(m*K)") == pytest.approx(1.5)


def test_rejects_unconverted_value() -> None:
    result = validate_conversion(0.015, "W/(cm*K)", 0.015, "W/(m*K)")
    assert not result.valid
    assert result.expected_value == pytest.approx(1.5)
    assert result.message == "unit conversion mismatch"


def test_auditor_rejects_unconverted_value() -> None:
    report = AuditorAgent().audit(make_parameter(0.015))
    assert report.status is WorkflowStatus.REJECTED
    assert "unit conversion mismatch" in report.failed_checks


def test_auditor_rejects_negative_value() -> None:
    report = AuditorAgent().audit(make_parameter(-1.5))
    assert report.status is WorkflowStatus.REJECTED
    assert "conductivity must be positive" in report.failed_checks

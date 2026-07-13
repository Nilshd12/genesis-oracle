from __future__ import annotations

import pytest

from collective_lab_validation.agents import AuditorAgent, ScholarPrime
from collective_lab_validation.models.parameter_payload import (
    ParameterPayload,
    SourceRecord,
    WorkflowStatus,
)


@pytest.fixture
def tensor_source() -> SourceRecord:
    return SourceRecord(
        title="Thermal conductivities of some novel nonlinear optical materials",
        authors=["J. Donald Beasley"],
        publication_year=1994,
        doi="10.1364/AO.33.001000",
        source_url="https://doi.org/10.1364/AO.33.001000",
        source_repository="test fixture",
        source_excerpt=(
            "AgGaS2 has 0.014 W/(cm K) parallel and 0.015 W/(cm K) "
            "perpendicular to the optic axis."
        ),
        parameter_names=["parallel", "perpendicular"],
        source_values={"parallel": 0.014, "perpendicular": 0.015},
        source_unit="W/(cm*K)",
        retrieval_mode="test",
    )


@pytest.fixture
def wrong_tensor_payload(tensor_source: SourceRecord) -> ParameterPayload:
    return ScholarPrime().extract(
        tensor_source,
        inject_reproducible_error=True,
    )


@pytest.fixture
def corrected_tensor_payload(
    wrong_tensor_payload: ParameterPayload,
) -> ParameterPayload:
    audit = AuditorAgent().audit(wrong_tensor_payload)
    return ScholarPrime().correct(wrong_tensor_payload, audit)


@pytest.fixture
def verified_tensor_payload(
    corrected_tensor_payload: ParameterPayload,
) -> ParameterPayload:
    return corrected_tensor_payload.model_copy(
        update={"validation_status": WorkflowStatus.VERIFIED}
    )

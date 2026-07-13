from __future__ import annotations

from pathlib import Path

from collective_lab_validation.agents.auditor_agent import AuditorAgent
from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.simulation.jax_heat_simulation import (
    run_heat_simulation,
)
from collective_lab_validation.validation.deterministic_gate import evaluate_gate


def test_jax_runs_after_approval_and_writes_temperature_image(
    corrected_tensor_payload: ParameterPayload,
    verified_tensor_payload: ParameterPayload,
    tmp_path: Path,
) -> None:
    audit = AuditorAgent().audit(corrected_tensor_payload)
    gate = evaluate_gate(verified_tensor_payload, audit)
    image_path = tmp_path / "heat.png"
    result = run_heat_simulation(
        verified_tensor_payload,
        gate,
        image_path=image_path,
        grid_size=15,
        steps=3,
    )
    assert result.completed
    assert result.conductivity_matrix == [[1.4, 0.0], [0.0, 1.5]]
    assert image_path.is_file()
    assert image_path.stat().st_size > 1_000

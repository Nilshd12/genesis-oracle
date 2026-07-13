from __future__ import annotations

from pathlib import Path

import pytest

from collective_lab_validation.models.gate_decision import GateDecision, GateStatus
from collective_lab_validation.models.parameter_payload import ParameterPayload
from collective_lab_validation.simulation.jax_heat_simulation import (
    run_heat_simulation,
)


def test_jax_refuses_to_start_without_gate_approval(
    verified_tensor_payload: ParameterPayload,
    tmp_path: Path,
) -> None:
    blocked = GateDecision(
        status=GateStatus.BLOCKED,
        execution_allowed=False,
        reasons=["rejected"],
    )
    with pytest.raises(PermissionError, match="simulation denied"):
        run_heat_simulation(
            verified_tensor_payload,
            blocked,
            image_path=tmp_path / "must_not_exist.png",
        )
    assert not (tmp_path / "must_not_exist.png").exists()

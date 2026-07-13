"""A small one-dimensional heat equation implemented with JAX."""

from __future__ import annotations

from collective_lab_demo.models.parameter_models import (
    GateDecision,
    GateStatus,
    ParameterRecord,
    SimulationResult,
    WorkflowStatus,
)
from collective_lab_demo.validation.gate import parameter_fingerprint
from collective_lab_demo.validation.unit_validator import (
    CANONICAL_METRE_UNIT,
    convert_value,
)


def _require_approval(
    parameter: ParameterRecord,
    decision: GateDecision,
) -> None:
    """Refuse stale, forged, blocked, or payload-mismatched approvals."""

    if (
        decision.status is not GateStatus.APPROVED
        or not decision.execution_allowed
        or parameter.validation_status is not WorkflowStatus.VERIFIED
        or decision.parameter_fingerprint != parameter_fingerprint(parameter)
    ):
        raise PermissionError(
            "JAX simulation denied: a matching deterministic gate approval "
            "for a VERIFIED parameter is required."
        )


def run_heat_simulation(
    parameter: ParameterRecord,
    decision: GateDecision,
    *,
    steps: int = 30,
    grid_points: int = 21,
) -> SimulationResult:
    """Evolve a stable 1D temperature profile after explicit gate approval."""

    _require_approval(parameter, decision)
    if steps < 1:
        raise ValueError("Simulation steps must be positive.")
    if grid_points < 5:
        raise ValueError("At least five grid points are required.")

    # Importing JAX is intentionally delayed until after the runtime gate.
    import jax
    import jax.numpy as jnp

    conductivity = convert_value(
        parameter.value,
        parameter.unit,
        CANONICAL_METRE_UNIT,
    )
    density = 1000.0  # kg/m^3, compact demonstrator material
    heat_capacity = 1000.0  # J/(kg*K)
    length = 1.0  # m
    dx = length / (grid_points - 1)
    diffusivity = conductivity / (density * heat_capacity)
    max_stable_dt = 0.45 * dx * dx / diffusivity
    dt = min(300.0, max_stable_dt)
    stability_number = diffusivity * dt / (dx * dx)

    initial = jnp.full((grid_points,), 20.0, dtype=jnp.float32)
    centre = grid_points // 2
    initial = initial.at[centre - 1 : centre + 2].set(100.0)

    def advance(_: int, temperature: jax.Array) -> jax.Array:
        interior = temperature[1:-1] + stability_number * (
            temperature[:-2]
            - 2.0 * temperature[1:-1]
            + temperature[2:]
        )
        updated = temperature.at[1:-1].set(interior)
        return updated.at[0].set(20.0).at[-1].set(20.0)

    final = jax.lax.fori_loop(0, steps, advance, initial)
    final.block_until_ready()
    initial_values = [float(value) for value in initial.tolist()]
    final_values = [float(value) for value in final.tolist()]

    return SimulationResult(
        thermal_conductivity=conductivity,
        unit=CANONICAL_METRE_UNIT,
        steps=steps,
        grid_points=grid_points,
        time_step_seconds=float(dt),
        stability_number=float(stability_number),
        initial_profile_celsius=initial_values,
        final_profile_celsius=final_values,
        minimum_temperature_celsius=min(final_values),
        maximum_temperature_celsius=max(final_values),
        mean_temperature_celsius=sum(final_values) / len(final_values),
        completed=True,
    )

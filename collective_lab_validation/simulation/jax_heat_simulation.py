"""Small two-dimensional anisotropic heat equation solved with JAX."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field

from collective_lab_validation.models.gate_decision import GateDecision, GateStatus
from collective_lab_validation.models.parameter_payload import (
    Matrix,
    ParameterPayload,
    StrictModel,
    WorkflowStatus,
    utc_now,
)
from collective_lab_validation.validation.deterministic_gate import (
    payload_fingerprint,
)


class SimulationResult(StrictModel):
    conductivity_matrix: Matrix
    unit: str
    grid_shape: tuple[int, int]
    steps: int
    time_step_seconds: float
    minimum_temperature_celsius: float
    maximum_temperature_celsius: float
    mean_temperature_celsius: float
    completed: bool
    result_image: str
    timestamp: datetime = Field(default_factory=utc_now)


def _require_approval(payload: ParameterPayload, gate: GateDecision) -> None:
    if (
        gate.status is not GateStatus.APPROVED
        or not gate.execution_allowed
        or payload.validation_status is not WorkflowStatus.VERIFIED
        or gate.payload_fingerprint != payload_fingerprint(payload)
    ):
        raise PermissionError(
            "JAX simulation denied: matching APPROVED gate and VERIFIED payload required."
        )


def run_heat_simulation(
    payload: ParameterPayload,
    gate: GateDecision,
    *,
    image_path: str | Path,
    grid_size: int = 51,
    steps: int = 40,
) -> SimulationResult:
    """Run JAX only after approval and save a headless temperature PNG."""

    _require_approval(payload, gate)
    if grid_size < 7 or steps < 1:
        raise ValueError("grid_size must be >= 7 and steps must be positive")

    # Delayed imports make the runtime guard precede all JAX execution.
    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conductivity = jnp.asarray(payload.matrix, dtype=jnp.float32)
    k_xx = conductivity[0, 0]
    k_xy = conductivity[0, 1]
    k_yy = conductivity[1, 1]
    density_heat_capacity = 1_000_000.0  # J/(m^3*K), compact demo material
    dx = 1.0 / (grid_size - 1)
    dy = dx
    denominator = (
        k_xx / (dx * dx)
        + k_yy / (dy * dy)
        + 2.0 * jnp.abs(k_xy) / (dx * dy)
    )
    dt = 0.2 * density_heat_capacity / denominator

    axis = jnp.linspace(0.0, 1.0, grid_size)
    x_grid, y_grid = jnp.meshgrid(axis, axis)
    initial = jnp.where(
        (x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2 <= 0.08**2,
        100.0,
        20.0,
    ).astype(jnp.float32)

    def advance(_: int, temperature: jax.Array) -> jax.Array:
        centre = temperature[1:-1, 1:-1]
        d2x = (
            temperature[1:-1, 2:]
            - 2.0 * centre
            + temperature[1:-1, :-2]
        ) / (dx * dx)
        d2y = (
            temperature[2:, 1:-1]
            - 2.0 * centre
            + temperature[:-2, 1:-1]
        ) / (dy * dy)
        dxy = (
            temperature[2:, 2:]
            - temperature[2:, :-2]
            - temperature[:-2, 2:]
            + temperature[:-2, :-2]
        ) / (4.0 * dx * dy)
        updated_centre = centre + dt * (
            k_xx * d2x + 2.0 * k_xy * dxy + k_yy * d2y
        ) / density_heat_capacity
        updated = temperature.at[1:-1, 1:-1].set(updated_centre)
        updated = updated.at[0, :].set(20.0).at[-1, :].set(20.0)
        return updated.at[:, 0].set(20.0).at[:, -1].set(20.0)

    final = jax.lax.fori_loop(0, steps, advance, initial)
    final.block_until_ready()
    values = final.tolist()
    flattened = [float(value) for row in values for value in row]

    target = Path(image_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis_plot = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    image = axis_plot.imshow(
        final,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="inferno",
        vmin=20.0,
        vmax=100.0,
    )
    axis_plot.set_title("Anisotrope JAX-Wärmesimulation nach VERIFIED-Gate")
    axis_plot.set_xlabel("x")
    axis_plot.set_ylabel("y")
    figure.colorbar(image, ax=axis_plot, label="Temperatur [°C]")
    figure.savefig(target, dpi=160)
    plt.close(figure)

    return SimulationResult(
        conductivity_matrix=payload.matrix,
        unit=payload.unit,
        grid_shape=(grid_size, grid_size),
        steps=steps,
        time_step_seconds=float(dt),
        minimum_temperature_celsius=min(flattened),
        maximum_temperature_celsius=max(flattened),
        mean_temperature_celsius=sum(flattened) / len(flattened),
        completed=True,
        result_image=str(target),
    )

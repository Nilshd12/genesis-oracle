"""Vectorized JAX escape-time calculation for the Mandelbrot set."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from fractal_entropy_demo.models.search_models import SearchBounds


@partial(jax.jit, static_argnames=("max_iterations",))
def _escape_kernel(c: jax.Array, max_iterations: int) -> jax.Array:
    """Compute iterations until ``abs(z) > 2`` for every point in parallel."""

    z = jnp.zeros_like(c)
    active = jnp.ones(c.shape, dtype=jnp.bool_)
    escape_times = jnp.full(c.shape, max_iterations, dtype=jnp.int32)

    def iteration_step(
        iteration: int,
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        current_z, current_active, current_times = state
        candidate = jnp.where(current_active, current_z * current_z + c, current_z)
        escaped_now = current_active & (jnp.abs(candidate) > 2.0)
        updated_times = jnp.where(escaped_now, iteration + 1, current_times)
        updated_active = current_active & ~escaped_now
        return candidate, updated_active, updated_times

    _, _, result = jax.lax.fori_loop(
        0,
        max_iterations,
        iteration_step,
        (z, active, escape_times),
    )
    return result


def mandelbrot_escape_times(
    bounds: SearchBounds,
    grid_width: int,
    grid_height: int,
    max_iterations: int,
) -> jax.Array:
    """Return a ``(height, width)`` escape-time matrix computed by JAX.

    Escape-time is the first iteration at which ``abs(z) > 2``. Points that
    do not escape receive ``max_iterations``. No Python loop iterates pixels.
    """

    if grid_width < 2 or grid_height < 2:
        raise ValueError("The Mandelbrot grid must be at least 2x2.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    x_axis = jnp.linspace(bounds.x_min, bounds.x_max, grid_width)
    y_axis = jnp.linspace(bounds.y_min, bounds.y_max, grid_height)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis)
    complex_grid = x_grid + 1j * y_grid
    result = _escape_kernel(complex_grid, max_iterations)
    result.block_until_ready()
    return result

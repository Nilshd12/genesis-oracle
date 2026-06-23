import jax
import jax.numpy as jnp


@jax.jit
def mandelbrot_kernel(c, max_iters):
    """
    Berechnet für jeden komplexen Punkt die Anzahl der Iterationen
    bis zum Verlassen des Mandelbrot-Radius.
    """

    def body_fn(val):
        z, count, active, iteration = val

        next_z = jnp.where(active, z**2 + c, z)
        next_active = active & (jnp.abs(next_z) <= 2.0)
        next_count = jnp.where(next_active, count + 1, count)

        return next_z, next_count, next_active, iteration + 1

    def cond_fn(val):
        _, _, active, iteration = val
        return jnp.any(active) & (iteration < max_iters)

    z = jnp.zeros_like(c)
    count = jnp.zeros_like(c, dtype=jnp.int32)
    active = jnp.ones_like(c, dtype=jnp.bool_)
    iteration = jnp.array(0, dtype=jnp.int32)

    _, final_counts, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (z, count, active, iteration),
    )

    return final_counts


def run_simulation(
    center_real: float,
    center_imag: float,
    zoom: float,
    resolution: int = 400,
    max_iterations: int = 500,
):
    """
    Führt eine JAX-beschleunigte Mandelbrot-Simulation aus
    und berechnet Shannon-Entropie sowie Boundary Complexity.
    """

    if zoom <= 0:
        raise ValueError("zoom muss größer als 0 sein.")

    if resolution <= 0:
        raise ValueError("resolution muss größer als 0 sein.")

    if max_iterations <= 0:
        raise ValueError("max_iterations muss größer als 0 sein.")

    width = resolution
    height = resolution

    real_axis = jnp.linspace(
        center_real - 1.5 / zoom,
        center_real + 1.5 / zoom,
        width,
    )

    imag_axis = jnp.linspace(
        center_imag - 1.5 / zoom,
        center_imag + 1.5 / zoom,
        height,
    )

    real_grid, imag_grid = jnp.meshgrid(real_axis, imag_axis)
    complex_grid = real_grid + 1j * imag_grid

    counts = mandelbrot_kernel(
        complex_grid.flatten(),
        max_iterations,
    )

    counts = counts.reshape((height, width))

    histogram, _ = jnp.histogram(counts, bins=20)
    histogram_probability = histogram / jnp.sum(histogram)

    safe_probability = jnp.where(
        histogram_probability > 0,
        histogram_probability,
        1.0,
    )

    entropy = -jnp.sum(
        histogram_probability * jnp.log(safe_probability)
    )

    boundary_pixels = jnp.sum(
        (counts > 0) & (counts < max_iterations)
    )

    boundary_ratio = boundary_pixels / (width * height)

    metrics = {
        "entropy": float(entropy),
        "boundary_complexity": float(boundary_ratio),
        "center_real": float(center_real),
        "center_imag": float(center_imag),
        "zoom": float(zoom),
        "max_iterations": int(max_iterations),
    }

    return counts, metrics


def simulate_mandelbrot(
    center_real: float,
    center_imag: float,
    zoom: float,
    max_iterations: int = 500,
) -> dict:
    """
    Tool-Wrapper für agentische Aufrufe.

    Gibt nur die Simulationsmetriken als Dictionary zurück.
    """

    _, metrics = run_simulation(
        center_real=center_real,
        center_imag=center_imag,
        zoom=zoom,
        max_iterations=max_iterations,
    )

    return metrics


if __name__ == "__main__":
    test_metrics = simulate_mandelbrot(
        center_real=-0.743643887,
        center_imag=0.131825254,
        zoom=15000.0,
        max_iterations=500,
    )

    print("Mandelbrot solver test:")
    print(test_metrics)
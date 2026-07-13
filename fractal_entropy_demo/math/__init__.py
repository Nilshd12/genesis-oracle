"""JAX Mandelbrot and local Shannon-entropy calculations."""

from fractal_entropy_demo.math.mandelbrot_jax import mandelbrot_escape_times
from fractal_entropy_demo.math.region_search import search_region
from fractal_entropy_demo.math.shannon_entropy import shannon_entropy

__all__ = ["mandelbrot_escape_times", "search_region", "shannon_entropy"]

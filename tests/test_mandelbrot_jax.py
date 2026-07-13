from __future__ import annotations

import numpy as np

from fractal_entropy_demo.math.mandelbrot_jax import mandelbrot_escape_times
from fractal_entropy_demo.models.search_models import SearchBounds


BOUNDS = SearchBounds(x_min=-2.0, x_max=1.0, y_min=-1.2, y_max=1.2)


def test_mandelbrot_result_has_expected_shape() -> None:
    result = mandelbrot_escape_times(BOUNDS, 24, 16, 30)
    assert result.shape == (16, 24)


def test_escape_times_are_finite_and_inside_valid_range() -> None:
    values = np.asarray(mandelbrot_escape_times(BOUNDS, 20, 18, 25))
    assert np.isfinite(values).all()
    assert values.min() >= 1
    assert values.max() <= 25

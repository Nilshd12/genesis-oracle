from __future__ import annotations

import jax.numpy as jnp
import pytest

from fractal_entropy_demo.math.shannon_entropy import shannon_entropy


def test_single_value_distribution_has_zero_entropy() -> None:
    assert shannon_entropy(jnp.array([7, 7, 7, 7]), 4) == pytest.approx(0.0)


def test_two_value_uniform_distribution_has_one_bit() -> None:
    assert shannon_entropy(jnp.array([0, 1, 0, 1]), 2) == pytest.approx(1.0)

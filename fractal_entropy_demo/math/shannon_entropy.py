"""Shannon entropy for discrete escape-time distributions."""

from __future__ import annotations

import jax.numpy as jnp


def shannon_entropy(values: jnp.ndarray, histogram_bins: int) -> float:
    """Return ``-sum(p * log2(p))`` in bits for a discrete histogram."""

    flattened = jnp.asarray(values).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Entropy requires at least one value.")
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be positive.")

    histogram, _ = jnp.histogram(flattened, bins=histogram_bins)
    probabilities = histogram / jnp.sum(histogram)
    terms = jnp.where(
        probabilities > 0,
        probabilities * jnp.log2(probabilities),
        0.0,
    )
    return float(-jnp.sum(terms))


def tile_entropies(
    escape_times: jnp.ndarray,
    *,
    tile_size: int,
    histogram_bins: int,
    max_iterations: int,
) -> tuple[jnp.ndarray, int, int]:
    """Calculate binned entropy for every non-overlapping full tile in JAX."""

    height, width = escape_times.shape
    tile_rows = height // tile_size
    tile_columns = width // tile_size
    if tile_rows == 0 or tile_columns == 0:
        raise ValueError("tile_size is larger than the escape-time matrix")

    trimmed = escape_times[
        : tile_rows * tile_size,
        : tile_columns * tile_size,
    ]
    tiles = (
        trimmed.reshape(tile_rows, tile_size, tile_columns, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(tile_rows * tile_columns, tile_size * tile_size)
    )

    # Escape times are integers in [1, max_iterations]. This maps them into
    # equally wide histogram bins without a Python loop over tiles or pixels.
    indices = jnp.floor(
        (tiles.astype(jnp.float32) - 1.0)
        * histogram_bins
        / max_iterations
    ).astype(jnp.int32)
    indices = jnp.clip(indices, 0, histogram_bins - 1)
    histograms = jnp.sum(
        jax_one_hot(indices, histogram_bins),
        axis=1,
    )
    probabilities = histograms / (tile_size * tile_size)
    entropy_terms = jnp.where(
        probabilities > 0,
        probabilities * jnp.log2(probabilities),
        0.0,
    )
    entropies = -jnp.sum(entropy_terms, axis=1)
    entropies.block_until_ready()
    return entropies.reshape(tile_rows, tile_columns), tile_rows, tile_columns


def jax_one_hot(indices: jnp.ndarray, classes: int) -> jnp.ndarray:
    """Small local one-hot helper, kept separate for straightforward testing."""

    return jnp.arange(classes) == indices[..., None]

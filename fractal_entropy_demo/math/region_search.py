"""Find the highest local tile entropy inside one explorer's bounds."""

from __future__ import annotations

import numpy as np

from fractal_entropy_demo.math.mandelbrot_jax import mandelbrot_escape_times
from fractal_entropy_demo.math.shannon_entropy import tile_entropies
from fractal_entropy_demo.models.search_models import (
    ExplorerConfig,
    TileBounds,
)


class RegionSearchOutcome:
    """In-memory numerical result used to build the JSON payload."""

    def __init__(
        self,
        *,
        best_tile_bounds: TileBounds,
        best_center_x: float,
        best_center_y: float,
        max_shannon_entropy: float,
        evaluated_tile_count: int,
        warnings: list[str],
    ) -> None:
        self.best_tile_bounds = best_tile_bounds
        self.best_center_x = best_center_x
        self.best_center_y = best_center_y
        self.max_shannon_entropy = max_shannon_entropy
        self.evaluated_tile_count = evaluated_tile_count
        self.warnings = warnings


def search_region(config: ExplorerConfig) -> RegionSearchOutcome:
    """Evaluate non-overlapping local tiles and select the maximum entropy."""

    matrix = mandelbrot_escape_times(
        config.search_bounds,
        config.grid_width,
        config.grid_height,
        config.max_iterations,
    )
    entropy_map, tile_rows, tile_columns = tile_entropies(
        matrix,
        tile_size=config.tile_size,
        histogram_bins=config.histogram_bins,
        max_iterations=config.max_iterations,
    )
    entropy_values = np.asarray(entropy_map)
    best_flat_index = int(np.argmax(entropy_values))
    best_row, best_column = np.unravel_index(
        best_flat_index,
        entropy_values.shape,
    )

    x_step = (
        config.search_bounds.x_max - config.search_bounds.x_min
    ) / (config.grid_width - 1)
    y_step = (
        config.search_bounds.y_max - config.search_bounds.y_min
    ) / (config.grid_height - 1)
    x_start_index = best_column * config.tile_size
    x_end_index = x_start_index + config.tile_size - 1
    y_start_index = best_row * config.tile_size
    y_end_index = y_start_index + config.tile_size - 1
    tile_bounds = TileBounds(
        x_min=config.search_bounds.x_min + x_start_index * x_step,
        x_max=config.search_bounds.x_min + x_end_index * x_step,
        y_min=config.search_bounds.y_min + y_start_index * y_step,
        y_max=config.search_bounds.y_min + y_end_index * y_step,
    )

    warnings: list[str] = []
    if config.grid_width % config.tile_size:
        warnings.append("Right-edge pixels outside complete tiles were ignored.")
    if config.grid_height % config.tile_size:
        warnings.append("Top-edge pixels outside complete tiles were ignored.")

    return RegionSearchOutcome(
        best_tile_bounds=tile_bounds,
        best_center_x=(tile_bounds.x_min + tile_bounds.x_max) / 2.0,
        best_center_y=(tile_bounds.y_min + tile_bounds.y_max) / 2.0,
        max_shannon_entropy=float(entropy_values[best_row, best_column]),
        evaluated_tile_count=tile_rows * tile_columns,
        warnings=warnings,
    )

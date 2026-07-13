"""Create a presentation-ready combined escape-time result graphic."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fractal_entropy_demo.math.mandelbrot_jax import mandelbrot_escape_times
from fractal_entropy_demo.models.search_models import (
    CommanderResult,
    ExplorerResult,
    SearchBounds,
)


def plot_results(
    results: Sequence[ExplorerResult],
    commander_result: CommanderResult,
    output_path: str | Path,
    *,
    grid_size: int,
    max_iterations: int,
) -> Path:
    """Plot the explored union, both regions/tiles, and the winning point."""

    if len(results) != 2:
        raise ValueError("Visualization requires exactly two Explorer results.")
    overall = SearchBounds(
        x_min=min(result.search_bounds.x_min for result in results),
        x_max=max(result.search_bounds.x_max for result in results),
        y_min=min(result.search_bounds.y_min for result in results),
        y_max=max(result.search_bounds.y_max for result in results),
    )
    escape_times = mandelbrot_escape_times(
        overall,
        grid_size,
        grid_size,
        max_iterations,
    )

    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(
        escape_times,
        origin="lower",
        extent=(overall.x_min, overall.x_max, overall.y_min, overall.y_max),
        cmap="magma",
        interpolation="nearest",
        aspect="auto",
    )
    colors = ("#39d5ff", "#ff9f43")
    for result, color in zip(results, colors, strict=True):
        bounds = result.search_bounds
        axis.add_patch(
            Rectangle(
                (bounds.x_min, bounds.y_min),
                bounds.x_max - bounds.x_min,
                bounds.y_max - bounds.y_min,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{result.worker_id} search area",
            )
        )
        tile = result.best_tile_bounds
        if tile is None:
            raise ValueError(f"Missing best tile for {result.worker_id}")
        axis.add_patch(
            Rectangle(
                (tile.x_min, tile.y_min),
                tile.x_max - tile.x_min,
                tile.y_max - tile.y_min,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
        )
        axis.scatter(
            [result.best_center_x],
            [result.best_center_y],
            color=color,
            edgecolor="black",
            s=65,
            zorder=5,
            label=(
                f"{result.worker_id}: H={result.max_shannon_entropy:.4f} bit"
            ),
        )

    axis.scatter(
        [commander_result.winning_center_x],
        [commander_result.winning_center_y],
        marker="*",
        color="#43ff64",
        edgecolor="black",
        linewidth=1.0,
        s=240,
        zorder=7,
        label=f"Winner: {commander_result.winning_worker}",
    )
    axis.set_title(
        "Fractal Entropy Lab: highest local entropy found in sampled grids"
    )
    axis.set_xlabel("Real axis")
    axis.set_ylabel("Imaginary axis")
    axis.legend(loc="upper right", framealpha=0.92, fontsize=9)
    figure.colorbar(image, ax=axis, label="Mandelbrot escape time")

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target

from __future__ import annotations

from pathlib import Path

from fractal_entropy_demo.models.search_models import SearchBounds
from fractal_entropy_demo.orchestration.parallel_runner import (
    execution_intervals_overlap,
    run_parallel_explorers,
)
from run_fractal_entropy_demo import build_explorer_configs


def test_two_explorers_run_in_parallel_with_overlapping_intervals(
    tmp_path: Path,
) -> None:
    configs = build_explorer_configs(
        artifacts_dir=tmp_path,
        grid_size=24,
        max_iterations=12,
        tile_size=6,
        histogram_bins=6,
        explorer_a_bounds=SearchBounds(
            x_min=-2.0, x_max=-0.5, y_min=-1.2, y_max=1.2
        ),
        explorer_b_bounds=SearchBounds(
            x_min=-0.5, x_max=1.0, y_min=-1.2, y_max=1.2
        ),
    )
    first, second = run_parallel_explorers(configs, output=lambda _: None)
    assert Path(first.result_file).is_file()
    assert Path(second.result_file).is_file()
    assert execution_intervals_overlap(first, second)

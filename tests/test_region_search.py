from __future__ import annotations

from pathlib import Path

from fractal_entropy_demo.agents.explorer import run_explorer
from fractal_entropy_demo.models.search_models import (
    ExplorerConfig,
    ExplorerResult,
    SearchBounds,
    SearchStatus,
)
from run_fractal_entropy_demo import build_explorer_configs


def test_explorers_receive_different_search_bounds(tmp_path: Path) -> None:
    configs = build_explorer_configs(
        artifacts_dir=tmp_path,
        grid_size=32,
        max_iterations=20,
        tile_size=8,
        histogram_bins=8,
        explorer_a_bounds=SearchBounds(
            x_min=-2.0, x_max=-0.5, y_min=-1.2, y_max=1.2
        ),
        explorer_b_bounds=SearchBounds(
            x_min=-0.5, x_max=1.0, y_min=-1.2, y_max=1.2
        ),
    )
    assert configs[0].search_bounds != configs[1].search_bounds
    assert configs[0].search_bounds.x_max == configs[1].search_bounds.x_min


def test_explorer_writes_valid_completed_json(tmp_path: Path) -> None:
    output_path = tmp_path / "explorer.json"
    config = ExplorerConfig(
        worker_id="explorer_test",
        search_bounds=SearchBounds(
            x_min=-2.0, x_max=-0.5, y_min=-1.2, y_max=1.2
        ),
        grid_width=32,
        grid_height=32,
        max_iterations=20,
        tile_size=8,
        histogram_bins=8,
        output_path=str(output_path),
    )
    run_explorer(config, announce=lambda _: None)
    restored = ExplorerResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert restored.status is SearchStatus.COMPLETED
    assert restored.best_tile_bounds is not None
    assert restored.max_shannon_entropy is not None
    assert restored.evaluated_tile_count == 16

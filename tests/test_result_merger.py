from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fractal_entropy_demo.models.search_models import (
    ExplorerResult,
    SearchBounds,
    SearchStatus,
    TileBounds,
)
from fractal_entropy_demo.orchestration.result_merger import merge_results


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def explorer_result(worker_id: str, entropy: float) -> ExplorerResult:
    return ExplorerResult(
        worker_id=worker_id,
        search_bounds=SearchBounds(
            x_min=-2.0 if worker_id == "explorer_a" else -0.5,
            x_max=-0.5 if worker_id == "explorer_a" else 1.0,
            y_min=-1.2,
            y_max=1.2,
        ),
        grid_shape=(32, 32),
        max_iterations=20,
        tile_size=8,
        histogram_bins=8,
        best_tile_bounds=TileBounds(x_min=-1, x_max=-0.8, y_min=0, y_max=0.2),
        best_center_x=-0.9,
        best_center_y=0.1,
        max_shannon_entropy=entropy,
        evaluated_tile_count=16,
        started_at=START,
        finished_at=START + timedelta(seconds=1),
        duration_seconds=1.0,
        status=SearchStatus.COMPLETED,
        result_file=f"{worker_id}.json",
    )


def merge(first: ExplorerResult, second: ExplorerResult):
    return merge_results(
        (first, second),
        started_at=START,
        finished_at=START + timedelta(seconds=1),
        parallel_execution_confirmed=True,
    )


def test_commander_accepts_exactly_two_complete_results() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        merge_results(
            [explorer_result("explorer_a", 1.0)],
            started_at=START,
            finished_at=START,
            parallel_execution_confirmed=False,
        )

    failed = explorer_result("explorer_b", 2.0).model_copy(
        update={
            "status": SearchStatus.FAILED,
            "best_center_x": None,
            "max_shannon_entropy": None,
        }
    )
    with pytest.raises(ValueError, match="Incomplete Explorer result"):
        merge(explorer_result("explorer_a", 1.0), failed)


def test_commander_selects_higher_entropy() -> None:
    result = merge(
        explorer_result("explorer_a", 2.0),
        explorer_result("explorer_b", 2.5),
    )
    assert result.winning_worker == "explorer_b"
    assert result.winning_entropy == pytest.approx(2.5)


def test_tie_breaking_uses_ascending_worker_id() -> None:
    result = merge(
        explorer_result("explorer_b", 2.0),
        explorer_result("explorer_a", 2.0),
    )
    assert result.winning_worker == "explorer_a"
    assert "ascending worker_id" in result.comparison_summary

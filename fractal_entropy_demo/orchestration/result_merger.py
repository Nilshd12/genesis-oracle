"""Validate two explorer payloads and choose a deterministic winner."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from fractal_entropy_demo.models.search_models import (
    CommanderResult,
    ExplorerResult,
    SearchStatus,
)


def _validate_completed(result: ExplorerResult) -> None:
    required_values = (
        result.best_tile_bounds,
        result.best_center_x,
        result.best_center_y,
        result.max_shannon_entropy,
    )
    if result.status is not SearchStatus.COMPLETED or any(
        value is None for value in required_values
    ):
        raise ValueError(f"Incomplete Explorer result: {result.worker_id}")


def merge_results(
    results: Sequence[ExplorerResult],
    *,
    started_at: datetime,
    finished_at: datetime,
    parallel_execution_confirmed: bool,
) -> CommanderResult:
    """Select maximum entropy; ties go to lexicographically smaller worker ID."""

    if len(results) != 2:
        raise ValueError("Commander accepts exactly two Explorer results.")
    if results[0].worker_id == results[1].worker_id:
        raise ValueError("Commander requires two distinct worker IDs.")
    for result in results:
        _validate_completed(result)

    ordered = sorted(
        results,
        key=lambda result: (
            -float(result.max_shannon_entropy),
            result.worker_id,
        ),
    )
    winner = ordered[0]
    tied = (
        float(results[0].max_shannon_entropy)
        == float(results[1].max_shannon_entropy)
    )
    tie_text = (
        " Equal entropy was resolved by ascending worker_id."
        if tied
        else " The greater entropy wins."
    )
    summary = (
        f"{results[0].worker_id}={float(results[0].max_shannon_entropy):.6f} bit; "
        f"{results[1].worker_id}={float(results[1].max_shannon_entropy):.6f} bit."
        f"{tie_text} Winner: {winner.worker_id}."
    )
    return CommanderResult(
        explorer_results=list(results),
        winning_worker=winner.worker_id,
        winning_center_x=float(winner.best_center_x),
        winning_center_y=float(winner.best_center_y),
        winning_entropy=float(winner.max_shannon_entropy),
        comparison_summary=summary,
        parallel_execution_confirmed=parallel_execution_confirmed,
        started_at=started_at,
        finished_at=finished_at,
        status=SearchStatus.COMPLETED,
    )

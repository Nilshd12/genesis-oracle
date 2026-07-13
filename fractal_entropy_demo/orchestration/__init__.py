"""Parallel local fallback and deterministic result merging."""

from fractal_entropy_demo.orchestration.parallel_runner import (
    run_parallel_explorers,
)
from fractal_entropy_demo.orchestration.result_merger import merge_results

__all__ = ["merge_results", "run_parallel_explorers"]

"""Start exactly two Explorer CLIs concurrently in independent processes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

from fractal_entropy_demo.models.search_models import (
    ExplorerConfig,
    ExplorerResult,
    SearchStatus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def execution_intervals_overlap(
    first: ExplorerResult,
    second: ExplorerResult,
) -> bool:
    """Whether both measured explorer lifetimes share a positive interval."""

    return max(first.started_at, second.started_at) < min(
        first.finished_at,
        second.finished_at,
    )


def _command(config: ExplorerConfig, synchronize_at: float) -> list[str]:
    bounds = config.search_bounds
    return [
        sys.executable,
        "-m",
        "fractal_entropy_demo.agents.explorer",
        "--worker-id",
        config.worker_id,
        "--x-min",
        str(bounds.x_min),
        "--x-max",
        str(bounds.x_max),
        "--y-min",
        str(bounds.y_min),
        "--y-max",
        str(bounds.y_max),
        "--grid-width",
        str(config.grid_width),
        "--grid-height",
        str(config.grid_height),
        "--max-iterations",
        str(config.max_iterations),
        "--tile-size",
        str(config.tile_size),
        "--histogram-bins",
        str(config.histogram_bins),
        "--output",
        config.output_path,
        "--synchronize-at",
        str(synchronize_at),
    ]


def _stream_process(
    process: subprocess.Popen[str],
    output: Callable[[str], None],
) -> int:
    if process.stdout is None:
        raise RuntimeError("Explorer stdout pipe was not created.")
    for line in process.stdout:
        output(line.rstrip())
    return process.wait()


def run_parallel_explorers(
    configs: Sequence[ExplorerConfig],
    *,
    output: Callable[[str], None] = print,
) -> tuple[ExplorerResult, ExplorerResult]:
    """Launch exactly two subprocesses before waiting for either result."""

    if len(configs) != 2:
        raise ValueError("The Commander must delegate to exactly two explorers.")
    if configs[0].worker_id == configs[1].worker_id:
        raise ValueError("Explorer worker IDs must be different.")
    if configs[0].search_bounds == configs[1].search_bounds:
        raise ValueError("Explorer search bounds must be different.")

    for config in configs:
        output_path = Path(config.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

    synchronize_at = time.time() + 0.75
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    processes = [
        subprocess.Popen(
            _command(config, synchronize_at),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        for config in configs
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        exit_codes = list(
            pool.map(
                lambda process: _stream_process(process, output),
                processes,
            )
        )
    if any(exit_codes):
        raise RuntimeError(f"Explorer subprocess failed: exit codes {exit_codes}")

    results = tuple(
        ExplorerResult.model_validate_json(
            Path(config.output_path).read_text(encoding="utf-8")
        )
        for config in configs
    )
    if any(result.status is not SearchStatus.COMPLETED for result in results):
        raise RuntimeError("At least one Explorer did not complete successfully.")
    if not execution_intervals_overlap(results[0], results[1]):
        raise RuntimeError("Explorer execution intervals did not overlap.")
    return results[0], results[1]

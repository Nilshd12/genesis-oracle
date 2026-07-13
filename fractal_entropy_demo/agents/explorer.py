"""Standalone JAX explorer CLI for one bounded Mandelbrot region."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from fractal_entropy_demo.math.region_search import search_region
from fractal_entropy_demo.models.search_models import (
    ExplorerConfig,
    ExplorerResult,
    SearchBounds,
    SearchStatus,
    utc_now,
)


def _write_result(result: ExplorerResult) -> Path:
    target = Path(result.result_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _wait_for_synchronized_start(epoch_seconds: float | None) -> None:
    if epoch_seconds is None:
        return
    while True:
        remaining = epoch_seconds - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.02))


def run_explorer(
    config: ExplorerConfig,
    *,
    announce: Callable[[str], None] = print,
    synchronize_at: float | None = None,
) -> ExplorerResult:
    """Run one region search, persist its JSON, and report real metrics."""

    started_at = utc_now()
    start_counter = time.perf_counter()
    result_path = str(Path(config.output_path).expanduser().resolve())
    label = config.worker_id.replace("_", " ").upper()
    announce(f"[{label}] Worker-ID: {config.worker_id}")
    announce(
        f"[{label}] Bereich: x=[{config.search_bounds.x_min:g}, "
        f"{config.search_bounds.x_max:g}], y=[{config.search_bounds.y_min:g}, "
        f"{config.search_bounds.y_max:g}]"
    )

    try:
        _wait_for_synchronized_start(synchronize_at)
        announce(f"[{label}] JAX-Berechnung gestartet ...")
        outcome = search_region(config)
        finished_at = utc_now()
        result = ExplorerResult(
            worker_id=config.worker_id,
            search_bounds=config.search_bounds,
            grid_shape=(config.grid_height, config.grid_width),
            max_iterations=config.max_iterations,
            tile_size=config.tile_size,
            histogram_bins=config.histogram_bins,
            best_tile_bounds=outcome.best_tile_bounds,
            best_center_x=outcome.best_center_x,
            best_center_y=outcome.best_center_y,
            max_shannon_entropy=outcome.max_shannon_entropy,
            evaluated_tile_count=outcome.evaluated_tile_count,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - start_counter,
            status=SearchStatus.COMPLETED,
            warnings=outcome.warnings,
            result_file=result_path,
        )
        _write_result(result)
        announce(f"[{label}] Untersuchte Kacheln: {result.evaluated_tile_count}")
        announce(
            f"[{label}] Beste Koordinate: "
            f"({result.best_center_x:.9f}, {result.best_center_y:.9f})"
        )
        announce(
            f"[{label}] Shannon-Entropie: "
            f"{result.max_shannon_entropy:.6f} Bit"
        )
        announce(f"[{label}] Ergebnisdatei: {result.result_file}")
        return result
    except Exception as error:
        finished_at = utc_now()
        failed = ExplorerResult(
            worker_id=config.worker_id,
            search_bounds=config.search_bounds,
            grid_shape=(config.grid_height, config.grid_width),
            max_iterations=config.max_iterations,
            tile_size=config.tile_size,
            histogram_bins=config.histogram_bins,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - start_counter,
            status=SearchStatus.FAILED,
            warnings=[f"{type(error).__name__}: {error}"],
            result_file=result_path,
        )
        _write_result(failed)
        announce(f"[{label}] FEHLER: {error}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore one Mandelbrot region with JAX and local entropy."
    )
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--grid-width", type=int, default=384)
    parser.add_argument("--grid-height", type=int, default=384)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--histogram-bins", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--synchronize-at",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExplorerConfig:
    return ExplorerConfig(
        worker_id=args.worker_id,
        search_bounds=SearchBounds(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        ),
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        max_iterations=args.max_iterations,
        tile_size=args.tile_size,
        histogram_bins=args.histogram_bins,
        output_path=args.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_explorer(
            config_from_args(args),
            synchronize_at=args.synchronize_at,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

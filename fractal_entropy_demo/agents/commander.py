"""Commander orchestration and merge-only CLI for native agent delegation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from fractal_entropy_demo.models.search_models import (
    CommanderResult,
    ExplorerConfig,
    ExplorerResult,
    utc_now,
)
from fractal_entropy_demo.orchestration.parallel_runner import (
    execution_intervals_overlap,
    run_parallel_explorers,
)
from fractal_entropy_demo.orchestration.result_merger import merge_results
from fractal_entropy_demo.visualization.plot_results import plot_results


def write_commander_result(result: CommanderResult, output_path: str | Path) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


class Commander:
    """Delegate to exactly two local explorer processes and merge their JSON."""

    def run(
        self,
        configs: Sequence[ExplorerConfig],
        *,
        output_path: str | Path,
        output: Callable[[str], None] = print,
    ) -> CommanderResult:
        started_at = utc_now()
        first, second = run_parallel_explorers(configs, output=output)
        finished_at = utc_now()
        merged = merge_results(
            (first, second),
            started_at=started_at,
            finished_at=finished_at,
            parallel_execution_confirmed=execution_intervals_overlap(first, second),
        )
        write_commander_result(merged, output_path)
        return merged

    def merge_existing(
        self,
        result_paths: Sequence[str | Path],
        *,
        output_path: str | Path,
    ) -> CommanderResult:
        """Merge two files produced by native Antigravity Explorer subagents."""

        if len(result_paths) != 2:
            raise ValueError("Exactly two Explorer result paths are required.")
        results = [
            ExplorerResult.model_validate_json(
                Path(path).expanduser().read_text(encoding="utf-8")
            )
            for path in result_paths
        ]
        started_at = min(result.started_at for result in results)
        finished_at = max(result.finished_at for result in results)
        merged = merge_results(
            results,
            started_at=started_at,
            finished_at=finished_at,
            parallel_execution_confirmed=execution_intervals_overlap(
                results[0], results[1]
            ),
        )
        write_commander_result(merged, output_path)
        return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge exactly two completed Fractal Explorer JSON files."
    )
    parser.add_argument("--explorer-results", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot")
    parser.add_argument("--plot-grid-size", type=int, default=384)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = Commander().merge_existing(
            args.explorer_results,
            output_path=args.output,
        )
        if args.plot:
            plot_results(
                result.explorer_results,
                result,
                args.plot,
                grid_size=args.plot_grid_size,
                max_iterations=max(
                    explorer.max_iterations for explorer in result.explorer_results
                ),
            )
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"Commander merge failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local fallback demo for Commander plus two parallel JAX Explorers."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from fractal_entropy_demo.agents.commander import Commander
from fractal_entropy_demo.models.search_models import (
    CommanderResult,
    ExplorerConfig,
    SearchBounds,
)
from fractal_entropy_demo.visualization.plot_results import plot_results


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class FractalDemoSummary:
    commander_result: CommanderResult
    commander_result_path: Path
    visualization_path: Path
    duration_seconds: float


def _pause(enabled: bool, output: Callable[[str], None]) -> None:
    if not enabled:
        return
    if sys.stdin.isatty():
        input("[PRAESENTATION] Enter fuer den naechsten Schritt ... ")
    else:
        output("[PRAESENTATION] Pause uebersprungen (nicht interaktiv).")


def build_explorer_configs(
    *,
    artifacts_dir: str | Path,
    grid_size: int,
    max_iterations: int,
    tile_size: int,
    histogram_bins: int,
    explorer_a_bounds: SearchBounds,
    explorer_b_bounds: SearchBounds,
) -> tuple[ExplorerConfig, ExplorerConfig]:
    artifact_path = Path(artifacts_dir).expanduser().resolve()
    return (
        ExplorerConfig(
            worker_id="explorer_a",
            search_bounds=explorer_a_bounds,
            grid_width=grid_size,
            grid_height=grid_size,
            max_iterations=max_iterations,
            tile_size=tile_size,
            histogram_bins=histogram_bins,
            output_path=str(artifact_path / "explorer_a_result.json"),
        ),
        ExplorerConfig(
            worker_id="explorer_b",
            search_bounds=explorer_b_bounds,
            grid_width=grid_size,
            grid_height=grid_size,
            max_iterations=max_iterations,
            tile_size=tile_size,
            histogram_bins=histogram_bins,
            output_path=str(artifact_path / "explorer_b_result.json"),
        ),
    )


def run_demo(
    *,
    artifacts_dir: str | Path = PROJECT_ROOT / "fractal_demo_artifacts",
    grid_size: int = 384,
    max_iterations: int = 180,
    tile_size: int = 32,
    histogram_bins: int = 32,
    explorer_a_bounds: SearchBounds | None = None,
    explorer_b_bounds: SearchBounds | None = None,
    presentation: bool = False,
    no_pause: bool = False,
    output: Callable[[str], None] = print,
) -> FractalDemoSummary:
    """Run the local subprocess fallback; this is not native agent tracing."""

    start_counter = time.perf_counter()
    artifact_path = Path(artifacts_dir).expanduser().resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)
    commander_path = artifact_path / "commander_result.json"
    visualization_path = artifact_path / "fractal_entropy_result.png"
    for stale in (commander_path, visualization_path):
        if stale.exists():
            stale.unlink()

    bounds_a = explorer_a_bounds or SearchBounds(
        x_min=-2.0,
        x_max=-0.5,
        y_min=-1.2,
        y_max=1.2,
    )
    bounds_b = explorer_b_bounds or SearchBounds(
        x_min=-0.5,
        x_max=1.0,
        y_min=-1.2,
        y_max=1.2,
    )
    configs = build_explorer_configs(
        artifacts_dir=artifact_path,
        grid_size=grid_size,
        max_iterations=max_iterations,
        tile_size=tile_size,
        histogram_bins=histogram_bins,
        explorer_a_bounds=bounds_a,
        explorer_b_bounds=bounds_b,
    )

    output("=" * 68)
    output("FRACTAL ENTROPY LAB – Commander mit parallelen Explorern")
    output("=" * 68)
    output("[HINWEIS] Lokaler paralleler Fallback; kein simuliertes Antigravity-Tracing.")
    output("[COMMANDER] Auftrag erhalten:")
    output("Regionen mit maximaler lokaler Shannon-Entropie suchen.")
    output("")
    output("[COMMANDER] Delegiere Region A an Explorer A.")
    output("[COMMANDER] Delegiere Region B an Explorer B.")
    _pause(presentation and not no_pause, output)

    commander_result = Commander().run(
        configs,
        output_path=commander_path,
        output=output,
    )
    first, second = commander_result.explorer_results
    output("")
    output("[COMMANDER] Beide Resultate empfangen.")
    output(
        f"[EXPLORER A] Ergebnis: ({first.best_center_x:.9f}, "
        f"{first.best_center_y:.9f}), H={first.max_shannon_entropy:.6f} Bit"
    )
    output(
        f"[EXPLORER B] Ergebnis: ({second.best_center_x:.9f}, "
        f"{second.best_center_y:.9f}), H={second.max_shannon_entropy:.6f} Bit"
    )
    output(
        f"[COMMANDER] Höchste gefundene Entropie: "
        f"{commander_result.winning_entropy:.6f} Bit"
    )
    output(f"[COMMANDER] Gewinner: {commander_result.winning_worker}")
    output(
        f"[COMMANDER] Koordinate: ({commander_result.winning_center_x:.9f}, "
        f"{commander_result.winning_center_y:.9f})"
    )
    output(
        "[COMMANDER] Parallele Ausführung bestätigt: "
        f"{commander_result.parallel_execution_confirmed}"
    )
    _pause(presentation and not no_pause, output)

    output("[VISUALISIERUNG] Erzeuge Escape-Time-Grafik ...")
    plot_path = plot_results(
        commander_result.explorer_results,
        commander_result,
        visualization_path,
        grid_size=grid_size,
        max_iterations=max_iterations,
    )
    duration = time.perf_counter() - start_counter
    output(f"[ARTEFAKT] {commander_path}")
    output(f"[ARTEFAKT] {plot_path}")
    output(f"[FERTIG] Gesamtlaufzeit: {duration:.3f} Sekunden")
    return FractalDemoSummary(
        commander_result=commander_result,
        commander_result_path=commander_path,
        visualization_path=plot_path,
        duration_seconds=duration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Fractal Entropy Lab.")
    parser.add_argument("--presentation", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--grid-size", type=int, default=384)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--histogram-bins", type=int, default=32)
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "fractal_demo_artifacts"),
    )
    parser.add_argument("--a-x-min", type=float, default=-2.0)
    parser.add_argument("--a-x-max", type=float, default=-0.5)
    parser.add_argument("--a-y-min", type=float, default=-1.2)
    parser.add_argument("--a-y-max", type=float, default=1.2)
    parser.add_argument("--b-x-min", type=float, default=-0.5)
    parser.add_argument("--b-x-max", type=float, default=1.0)
    parser.add_argument("--b-y-min", type=float, default=-1.2)
    parser.add_argument("--b-y-max", type=float, default=1.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_demo(
            artifacts_dir=args.artifacts_dir,
            grid_size=args.grid_size,
            max_iterations=args.max_iterations,
            tile_size=args.tile_size,
            histogram_bins=args.histogram_bins,
            explorer_a_bounds=SearchBounds(
                x_min=args.a_x_min,
                x_max=args.a_x_max,
                y_min=args.a_y_min,
                y_max=args.a_y_max,
            ),
            explorer_b_bounds=SearchBounds(
                x_min=args.b_x_min,
                x_max=args.b_x_max,
                y_min=args.b_y_min,
                y_max=args.b_y_max,
            ),
            presentation=args.presentation,
            no_pause=args.no_pause,
        )
    except Exception as error:
        print(f"FRACTAL ENTROPY LAB fehlgeschlagen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

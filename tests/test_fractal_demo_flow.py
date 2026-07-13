from __future__ import annotations

import json
from pathlib import Path

from run_fractal_entropy_demo import run_demo


def test_complete_fractal_demo_creates_json_and_visualization(
    tmp_path: Path,
) -> None:
    messages: list[str] = []
    summary = run_demo(
        artifacts_dir=tmp_path,
        grid_size=48,
        max_iterations=24,
        tile_size=8,
        histogram_bins=8,
        presentation=True,
        no_pause=True,
        output=messages.append,
    )
    result = summary.commander_result
    assert result.parallel_execution_confirmed
    assert len(result.explorer_results) == 2
    assert result.winning_worker in {"explorer_a", "explorer_b"}
    assert summary.commander_result_path.is_file()
    assert summary.visualization_path.is_file()
    assert summary.visualization_path.stat().st_size > 1_000
    assert (tmp_path / "explorer_a_result.json").is_file()
    assert (tmp_path / "explorer_b_result.json").is_file()

    payload = json.loads(summary.commander_result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETED"
    assert payload["parallel_execution_confirmed"] is True
    assert "[COMMANDER] Beide Resultate empfangen." in "\n".join(messages)

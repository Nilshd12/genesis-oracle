"""Adapter for the installed arXiv Science Skill CLI (optional fallback)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARXIV_SCRIPT = (
    PROJECT_ROOT
    / "science-skills"
    / "skills"
    / "literature_search_arxiv"
    / "scripts"
    / "search_arxiv.py"
)


class ArxivAdapter:
    """Run the real installed arXiv skill without claiming source coverage."""

    skill_name = "literature-search-arxiv"

    def search(self, query: str, max_results: int = 3) -> dict[str, Any]:
        result = subprocess.run(
            [
                "uv",
                "run",
                str(ARXIV_SCRIPT),
                "--query",
                query,
                "--max_results",
                str(max_results),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "arXiv skill failed")
        return json.loads(result.stdout)

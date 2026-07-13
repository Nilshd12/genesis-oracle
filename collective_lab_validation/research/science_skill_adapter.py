"""Preferred DeepMind OpenAlex Science Skill with transparent fallbacks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from collective_lab_validation.models.parameter_payload import SourceRecord
from collective_lab_validation.research.openalex_adapter import (
    OpenAlexAdapter,
    SELECT_FIELDS,
    SOURCE_DOI_URL,
    source_from_openalex,
)
from collective_lab_validation.research.source_cache import load_source_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENALEX_SKILL_SCRIPT = (
    PROJECT_ROOT
    / "science-skills"
    / "skills"
    / "literature_search_openalex"
    / "scripts"
    / "openalex_cli.py"
)
SCIENCE_SKILL_NAME = "literature-search-openalex"


@dataclass(frozen=True, slots=True)
class ResearchResult:
    source: SourceRecord
    requested_mode: str
    actual_mode: str
    fallback_reasons: tuple[str, ...] = ()


class OpenAlexScienceSkillAdapter:
    """Invoke the installed Google DeepMind Science Skill exactly via uv."""

    skill_name = SCIENCE_SKILL_NAME

    def fetch_known_source(self) -> SourceRecord:
        if not OPENALEX_SKILL_SCRIPT.is_file():
            raise FileNotFoundError(f"Science Skill script missing: {OPENALEX_SKILL_SCRIPT}")
        result = subprocess.run(
            [
                "uv",
                "run",
                str(OPENALEX_SKILL_SCRIPT),
                "get",
                "works",
                SOURCE_DOI_URL,
                "--select",
                SELECT_FIELDS,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"OpenAlex Science Skill failed: {detail}")
        data = json.loads(result.stdout)
        return source_from_openalex(
            data,
            retrieval_mode="online-deepmind-science-skill",
            science_skill_used=self.skill_name,
        )


def research_source(mode: str) -> ResearchResult:
    if mode == "demo":
        return ResearchResult(
            source=load_source_fixture(),
            requested_mode=mode,
            actual_mode="fixture",
        )
    if mode != "online":
        raise ValueError("mode must be 'demo' or 'online'")

    reasons: list[str] = []
    try:
        source = OpenAlexScienceSkillAdapter().fetch_known_source()
        return ResearchResult(source, mode, "science-skill")
    except (OSError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        reasons.append(str(error))
    try:
        source = OpenAlexAdapter().fetch_known_source()
        return ResearchResult(source, mode, "openalex-api-fallback", tuple(reasons))
    except (OSError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        reasons.append(str(error))
    return ResearchResult(
        load_source_fixture(),
        mode,
        "fixture-fallback",
        tuple(reasons),
    )

"""Controlled, human-readable JSON output for each demo run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ArtifactService:
    """Write only known demo artifacts into a caller-selected directory."""

    GENERATED_FILES = frozenset(
        {
            "source_record.json",
            "extracted_parameters.json",
            "rejected_audit_report.json",
            "blocked_gate_decision.json",
            "corrected_parameters.json",
            "verified_audit_report.json",
            "gate_decision.json",
            "simulation_result.json",
        }
    )

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def prepare_run(self) -> None:
        """Remove only stale, explicitly named outputs from an earlier run."""

        for name in self.GENERATED_FILES:
            candidate = self.directory / name
            if candidate.is_file():
                candidate.unlink()

    def write_json(self, name: str, payload: BaseModel | Any) -> Path:
        """Serialize a model/dict using UTF-8 and stable indentation."""

        if name not in self.GENERATED_FILES:
            raise ValueError(f"Unknown Collective Lab artifact: {name}")
        data = (
            payload.model_dump(mode="json")
            if isinstance(payload, BaseModel)
            else payload
        )
        target = self.directory / name
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

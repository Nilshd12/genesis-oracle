"""Local verified source fixture used only in reproducible/fallback mode."""

from __future__ import annotations

from pathlib import Path

from collective_lab_validation.models.parameter_payload import SourceRecord


DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "verified_source_fixture.json"
)


def load_source_fixture(path: str | Path = DEFAULT_FIXTURE) -> SourceRecord:
    return SourceRecord.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )

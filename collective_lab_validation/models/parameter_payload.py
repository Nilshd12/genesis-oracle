"""Source and conductivity-tensor payload models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


Number = StrictFloat | StrictInt
Matrix = list[list[Number]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"
    VERIFIED = "VERIFIED"


class SourceRecord(StrictModel):
    title: str
    authors: list[str]
    publication_year: int
    doi: str
    source_url: str
    source_repository: str
    source_excerpt: str
    parameter_names: list[str]
    source_values: dict[str, Number]
    source_unit: str
    retrieved_at: datetime = Field(default_factory=utc_now)
    retrieval_mode: str
    science_skill_used: str | None = None


class CorrectionEntry(StrictModel):
    previous_matrix: Matrix
    previous_unit: str
    corrected_matrix: Matrix
    corrected_unit: str
    instruction: str
    timestamp: datetime = Field(default_factory=utc_now)


class ParameterPayload(StrictModel):
    parameter_name: str
    matrix: Matrix
    unit: str
    source_values: dict[str, Number]
    source_unit: str
    source_title: str
    source_doi: str
    source_url: str
    source_excerpt: str
    extraction_status: WorkflowStatus = WorkflowStatus.PENDING
    validation_status: WorkflowStatus = WorkflowStatus.PENDING
    correction_history: list[CorrectionEntry] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)

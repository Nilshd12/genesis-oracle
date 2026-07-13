"""Strict, serializable models shared by all Collective Lab components."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for reproducible JSON schemas."""

    return datetime.now(timezone.utc)


class WorkflowStatus(str, Enum):
    """Lifecycle states used by extraction and validation."""

    PENDING = "PENDING"
    REJECTED = "REJECTED"
    CORRECTED = "CORRECTED"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"


class GateStatus(str, Enum):
    """The two possible deterministic gate outcomes."""

    BLOCKED = "BLOCKED"
    APPROVED = "APPROVED"


class StrictModel(BaseModel):
    """Base model that rejects undocumented fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SourceRecord(StrictModel):
    """A traceable scientific source, obtained online or from a fixture."""

    title: str
    authors: list[str]
    publication_year: int
    url: str
    doi: str | None = None
    parameter_name: str
    original_value: float
    original_unit: str
    excerpt: str
    provider: str
    retrieval_mode: str
    is_fixture: bool
    retrieved_at: datetime = Field(default_factory=utc_now)


class CorrectionRecord(StrictModel):
    """One explicit correction made in response to an audit."""

    previous_value: float
    previous_unit: str
    corrected_value: float
    corrected_unit: str
    reason: str
    timestamp: datetime = Field(default_factory=utc_now)


class ParameterRecord(StrictModel):
    """Scholar-Prime's structured parameter payload."""

    parameter_name: str
    value: float
    unit: str
    source_value: float
    source_unit: str
    source_title: str
    source_url: str
    source_excerpt: str
    extraction_status: WorkflowStatus = WorkflowStatus.PENDING
    validation_status: WorkflowStatus = WorkflowStatus.PENDING
    correction_history: list[CorrectionRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)


class AuditReport(StrictModel):
    """Structured output from the independent Auditor-Agent."""

    status: WorkflowStatus
    passed_checks: list[str]
    failed_checks: list[str]
    expected_value: float | None = None
    received_value: float | None = None
    expected_unit: str | None = None
    received_unit: str | None = None
    correction_instruction: str | None = None
    audit_summary: str
    timestamp: datetime = Field(default_factory=utc_now)


class GateDecision(StrictModel):
    """Non-LLM execution decision bound to one exact parameter payload."""

    status: GateStatus
    execution_allowed: bool
    reasons: list[str]
    parameter_fingerprint: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class SimulationResult(StrictModel):
    """Compact, JSON-friendly output of the JAX heat simulation."""

    thermal_conductivity: float
    unit: str
    steps: int
    grid_points: int
    time_step_seconds: float
    stability_number: float
    initial_profile_celsius: list[float]
    final_profile_celsius: list[float]
    minimum_temperature_celsius: float
    maximum_temperature_celsius: float
    mean_temperature_celsius: float
    completed: bool
    timestamp: datetime = Field(default_factory=utc_now)

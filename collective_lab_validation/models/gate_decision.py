"""Deterministic execution-gate result."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from collective_lab_validation.models.parameter_payload import StrictModel, utc_now


class GateStatus(str, Enum):
    BLOCKED = "BLOCKED"
    APPROVED = "APPROVED"


class GateDecision(StrictModel):
    status: GateStatus
    execution_allowed: bool
    reasons: list[str]
    payload_fingerprint: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)

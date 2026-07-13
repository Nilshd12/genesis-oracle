"""Strict models shared across processes and native agent prompts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SearchStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SearchBounds(StrictModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def validate_order(self) -> "SearchBounds":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be smaller than y_max")
        return self


class TileBounds(StrictModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class ExplorerConfig(StrictModel):
    worker_id: str
    search_bounds: SearchBounds
    grid_width: int = Field(ge=8)
    grid_height: int = Field(ge=8)
    max_iterations: int = Field(ge=2)
    tile_size: int = Field(ge=2)
    histogram_bins: int = Field(ge=2)
    output_path: str

    @model_validator(mode="after")
    def validate_tile_size(self) -> "ExplorerConfig":
        if self.tile_size > min(self.grid_width, self.grid_height):
            raise ValueError("tile_size must fit inside the grid")
        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not self.output_path.strip():
            raise ValueError("output_path must not be empty")
        return self


class ExplorerResult(StrictModel):
    worker_id: str
    search_bounds: SearchBounds
    grid_shape: tuple[int, int]
    max_iterations: int
    tile_size: int
    histogram_bins: int
    best_tile_bounds: TileBounds | None = None
    best_center_x: float | None = None
    best_center_y: float | None = None
    max_shannon_entropy: float | None = None
    evaluated_tile_count: int = 0
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0.0)
    status: SearchStatus
    warnings: list[str] = Field(default_factory=list)
    result_file: str


class CommanderResult(StrictModel):
    explorer_results: list[ExplorerResult]
    winning_worker: str
    winning_center_x: float
    winning_center_y: float
    winning_entropy: float
    comparison_summary: str
    parallel_execution_confirmed: bool
    started_at: datetime
    finished_at: datetime
    status: SearchStatus

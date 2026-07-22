"""Job step execution models for Solar Control job routing."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_host(cls, value: str | None) -> "JobStatus | None":
        """Map a Solar Host status string to a :class:`JobStatus`.

        Returns ``None`` for unknown/empty values so callers can decide
        whether to leave the current status untouched.
        """
        if not value:
            return None
        try:
            return cls(value.lower())
        except ValueError:
            return None


class Job(BaseModel):
    """A job step routed through Solar Control to a Solar Host."""

    id: str
    host_id: str
    status: JobStatus = JobStatus.PENDING
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="The translated host-level job config sent to the host",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Response payload from the host on completion",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the job failed or submission failed",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for matching S-025/S-026 events to this job",
    )
    submission_id: str | None = Field(
        default=None,
        description="Optional SuperNova submission ID, forwarded to the host",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class JobCreate(BaseModel):
    """Request payload for submitting a new job step.

    This is the SuperNova-level job intent. Solar Control translates
    this into the host-level job config (per S-021 workspace spec)
    before proxying to the selected Solar Host.

    Expected payload shape (see ``_translate_payload`` in
    ``app.jobs.router`` for the full schema):

    .. code-block:: python

        {
            "name": "job-name",
            "pipeline": ["download_model", "train", ...],
            "base_model_uri": "repo://...",
            "training_data_uri": "repo://...",
            "training_config": { ... },
            "model_selection": {"strategy": "best_metric", ...},
            "deployment": {"target": "...", "replicas": 2, ...},
            "retention_hours": 24,
            "steps": {
                "download_model": {"model_uri": "...", "output_dir": "..."},
                "train": {"run_name": "...", "output_dir": "...", "wandb": false},
                ...
            }
        }
    """

    payload: dict[str, Any] = Field(
        ...,
        description="SuperNova-level job step configuration (see S-021 spec)",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Optional correlation ID for event matching",
    )
    submission_id: str | None = Field(
        default=None,
        description="Optional SuperNova submission ID, forwarded to the host",
    )


class JobResponse(BaseModel):
    """Response returned after job submission."""

    job: Job
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    job: Job
    host_status: dict[str, Any] | None = Field(
        default=None,
        description="Real-time status from the host, if available",
    )

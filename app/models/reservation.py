"""Pydantic models for resource reservations (S-038)."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ReservationRequest(BaseModel):
    """Request to reserve resources on a Solar Host, routed through Solar Control."""

    # Required identity
    requester: str = Field(
        ..., description="Caller identifier (e.g. 'supernova', 'reconciler')"
    )
    job_id: str = Field(
        ..., description="Unique job/request identifier from the caller"
    )

    # Resources requested
    vram_gb: float = Field(..., gt=0, description="Requested VRAM in GB")
    ram_gb: float | None = Field(default=None, gt=0, description="Requested RAM in GB")
    disk_gb: float | None = Field(
        default=None, gt=0, description="Requested disk in GB"
    )

    # Workload & constraints
    workload_type: str = Field(
        default="training",
        description="Type of workload (e.g. 'training', 'inference')",
    )
    priority: str = Field(
        default="staging",
        description="Priority: production, staging, or ephemeral",
    )
    host_roles: list[str] = Field(
        default_factory=lambda: ["training"],
        description="Required host roles (host must have all listed)",
    )
    gpu_type: str | None = Field(
        default=None, description="Required GPU type (null means any)"
    )
    host_allow: list[str] = Field(
        default_factory=list,
        description="If non-empty, restrict to these host IDs",
    )
    host_deny: list[str] = Field(
        default_factory=list, description="Host IDs to exclude"
    )

    # Duration
    ttl_seconds: int | None = Field(
        default=None, gt=0, description="Optional TTL in seconds; host-side expiry"
    )
    expiration: str | None = Field(
        default=None,
        description="ISO 8601 expiration timestamp (alternative to ttl_seconds)",
    )

    # Preserve-one-replica constraint
    preserve_alias: str | None = Field(
        default=None,
        description=(
            "When evaluating hosts for migration, preserve at least one replica "
            "of this alias per host. Used to enforce the one-replica-per-host "
            "rule during displacement."
        ),
    )


class ReservationResponse(BaseModel):
    """Response returned after a successful reservation."""

    reservation_id: str = Field(..., description="Solar Control reservation ID")
    host_reservation_id: str = Field(..., description="Host-level reservation ID")
    host_id: str
    host_name: str
    host_url: str
    vram_gb: float
    ram_gb: float | None = None
    disk_gb: float | None = None
    workload_type: str
    priority: str
    expiration: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    migrated: bool = Field(
        default=False,
        description=("Whether lower-priority workloads were migrated to free capacity"),
    )
    migrations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Details of any migrations performed",
    )


class ReservationFailure(BaseModel):
    """Deterministic failure reason when capacity cannot be reserved."""

    reason: str = Field(
        ...,
        description=(
            "Machine-readable failure code "
            "(e.g. 'no_eligible_host', 'insufficient_capacity')"
        ),
    )
    detail: str = Field(..., description="Human-readable explanation")
    requested: ReservationRequest | None = None
    eligible_hosts: int = 0
    hosts_checked: int = 0
    migration_candidates: int = Field(
        default=0,
        description="Number of lower-priority instances that could be migrated",
    )


class ReservationReleaseResponse(BaseModel):
    """Response returned after releasing a reservation."""

    reservation_id: str
    host_reservation_id: str
    host_id: str
    released: bool
    message: str


class MigrationCandidate(BaseModel):
    """A lower-priority instance that could be migrated to free capacity."""

    host_id: str
    host_name: str
    instance_id: str
    alias: str
    priority: str
    vram_gb: float

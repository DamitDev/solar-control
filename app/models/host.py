from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class HostStatus(str, Enum):
    """Status of a solar-host"""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"


class MemoryInfo(BaseModel):
    """Memory usage information"""

    used_gb: float = Field(..., description="Used memory in GB")
    total_gb: float = Field(..., description="Total memory in GB")
    available_gb: float | None = Field(
        default=None,
        description="Memory available for new workloads (total - used)",
    )
    percent: float = Field(..., description="Usage percentage")
    memory_type: str = Field(..., description="Type of memory (VRAM or RAM)")


class Host(BaseModel):
    """Solar host information"""

    id: str
    name: str
    url: str
    api_key: str
    status: HostStatus = HostStatus.OFFLINE
    last_seen: datetime | None = None
    memory: MemoryInfo | None = None
    gpu_type: str | None = None
    roles: list[str] = Field(default_factory=list)
    disk_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_available_gb: float | None = None
    memory_available_gb: float | None = None
    version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HostCreate(BaseModel):
    """Request to register a new host"""

    name: str
    url: str
    api_key: str


class HostResponse(BaseModel):
    """Response for host operations"""

    host: Host
    message: str


class ActiveJobSummary(BaseModel):
    """Summary of an active or recently-terminal job on a host.

    Provides operators and scheduling logic with a lightweight view of
    what job workloads a host is running, without exposing the full
    ``Job`` payload.
    """

    job_id: str
    submission_id: str | None = Field(
        default=None,
        description="SuperNova submission ID for cross-system correlation",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable job name from the pipeline definition",
    )
    status: str = Field(
        ...,
        description="Current job status (pending/running/completed/failed/cancelled)",
    )
    current_step_name: str | None = Field(
        default=None,
        description=(
            "Name of the currently executing pipeline step (e.g. 'train'). "
            "Only set while the job is pending or running"
        ),
    )
    current_step_index: int | None = Field(
        default=None,
        description=(
            "Zero-based index of the currently executing pipeline step. "
            "Only set while the job is pending or running"
        ),
    )
    last_step_name: str | None = Field(
        default=None,
        description=(
            "Name of the last pipeline step the job entered. Remains set "
            "after the job reaches a terminal state, so consumers can see "
            "which step a failed job stopped on"
        ),
    )
    last_step_index: int | None = Field(
        default=None,
        description="Zero-based index of the last pipeline step the job entered",
    )
    pipeline: list[str] = Field(
        default_factory=list,
        description="Ordered list of all pipeline step names",
    )
    resource_hints: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Resource requirements extracted from the translated job "
            "definition (peak GPU count, minimum free disk, training config)"
        ),
    )
    started_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the job was created or started",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the job reached a terminal state",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the job failed or was cancelled",
    )


class HostResourceSnapshot(BaseModel):
    """Per-host resource snapshot in the aggregated cluster view (S-035).

    Combines locally stored metadata (roles, GPU type) with live resource
    data proxied from the solar-host ``GET /resources`` endpoint.

    Resource availability follows S-034 semantics:
    ``available = total - (system_used + reserved_headroom)`` where
    ``reserved_headroom = Σ max(reserved − actual, 0)`` per reservation.
    """

    host_id: str
    host_name: str
    url: str
    status: HostStatus
    roles: list[str] = Field(default_factory=list)
    gpu_type: str | None = None
    version: str | None = None

    # Whether the host was reachable for live resource data
    reachable: bool = False
    error: str | None = None

    # Total resources (from hardware)
    vram_total_gb: float | None = None
    ram_total_gb: float | None = None
    disk_total_gb: float | None = None

    # System-level usage (OS + idle backends)
    vram_system_used_gb: float | None = None
    ram_system_used_gb: float | None = None
    disk_system_used_gb: float | None = None

    # Reservation headroom (Σ max(reserved - actual, 0))
    vram_reserved_headroom_gb: float | None = None
    ram_reserved_headroom_gb: float | None = None
    disk_reserved_headroom_gb: float | None = None

    # Reported used = system_used + reserved_headroom
    vram_reported_used_gb: float | None = None
    ram_reported_used_gb: float | None = None
    disk_reported_used_gb: float | None = None

    # Available = total - reported_used
    vram_available_gb: float | None = None
    ram_available_gb: float | None = None
    disk_available_gb: float | None = None

    # Running workloads (from Redis instance cache)
    instance_count: int = 0
    running_instance_count: int = 0

    # Active job workloads (from jobs table — training pipelines etc.)
    active_jobs: list[ActiveJobSummary] = Field(
        default_factory=list,
        description="Active and recently-terminal job workloads on this host",
    )

    # Reservation summary (totals only — no per-reservation list)
    reservation_count: int = 0
    reservation_vram_total_gb: float = 0.0
    reservation_ram_total_gb: float = 0.0
    reservation_disk_total_gb: float = 0.0

    # Timestamps
    snapshot_timestamp: str | None = None


class AggregatedResourceResponse(BaseModel):
    """Aggregated cluster-wide resource view (S-035)."""

    hosts: list[HostResourceSnapshot]
    total_hosts: int
    reachable_hosts: int
    unreachable_hosts: int

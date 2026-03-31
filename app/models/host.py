from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum


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

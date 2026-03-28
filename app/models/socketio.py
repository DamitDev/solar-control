from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
from enum import Enum

from .host import Host, MemoryInfo


class WSMessageType(str, Enum):
    """WebSocket message types for the unified protocol"""

    REGISTRATION = "registration"
    LOG = "log"
    INSTANCE_STATE = "instance_state"
    HOST_HEALTH = "host_health"
    INSTANCES_UPDATE = "instances_update"

    HOST_STATUS = "host_status"
    INITIAL_STATUS = "initial_status"
    REQUEST_START = "request_start"
    REQUEST_ROUTED = "request_routed"
    REQUEST_SUCCESS = "request_success"
    REQUEST_ERROR = "request_error"
    REQUEST_REROUTE = "request_reroute"
    KEEPALIVE = "keepalive"


class WSMessage(BaseModel):
    """Base WebSocket message envelope"""

    type: WSMessageType
    host_id: str | None = None
    instance_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class WSRegistration(BaseModel):
    """Host registration message data."""

    api_key: str
    host_name: str | None = None
    instances: list[dict[str, Any]] = Field(default_factory=list)
    gpu_type: str | None = None
    roles: list[str] = Field(default_factory=list)


class WSLogMessage(BaseModel):
    """Log message from host"""

    seq: int
    line: str
    level: str | None = None


class WSInstanceState(BaseModel):
    """Instance runtime state update"""

    busy: bool = False
    phase: str | None = None
    prefill_progress: float | None = None
    active_slots: int = 0
    slot_id: int | None = None
    task_id: int | None = None
    prefill_prompt_tokens: int | None = None
    generated_tokens: int | None = None
    decode_tps: float | None = None
    decode_ms_per_token: float | None = None
    checkpoint_index: int | None = None
    checkpoint_total: int | None = None


class WSHostHealth(BaseModel):
    """Host health/memory update"""

    memory: MemoryInfo | None = None
    gpu_type: str | None = None
    instance_count: int = 0
    running_instance_count: int = 0
    disk_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_available_gb: float | None = None


class HostStatusPayload(BaseModel):
    """Outgoing payload for host_status events to WebUI."""

    host_id: str
    name: str | None
    status: str
    url: str | None
    last_seen: str | None = None
    memory: dict[str, Any] | None = None
    gpu_type: str | None = None
    roles: list[str] = Field(default_factory=list)
    disk_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_available_gb: float | None = None
    memory_available_gb: float | None = None
    connected: bool = False
    timestamp: str = ""

    @classmethod
    def from_host(cls, host: Host, *, connected: bool) -> "HostStatusPayload":
        return cls(
            host_id=host.id,
            name=host.name,
            status=host.status.value,
            url=host.url,
            last_seen=host.last_seen.isoformat() if host.last_seen else None,
            memory=host.memory.model_dump() if host.memory else None,
            gpu_type=host.gpu_type,
            roles=host.roles,
            disk_total_gb=host.disk_total_gb,
            disk_used_gb=host.disk_used_gb,
            disk_available_gb=host.disk_available_gb,
            memory_available_gb=host.memory_available_gb,
            connected=connected,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class InstancesUpdatePayload(BaseModel):
    """Outgoing payload for instances_update events to WebUI."""

    host_id: str
    instances: list[dict[str, Any]]


class HostPendingPayload(BaseModel):
    """Outgoing payload for host_pending events to WebUI."""

    pending_id: str
    api_key_preview: str
    host_name: str
    instance_count: int | None = None
    connected_at: str | None = None


class HostHealthPayload(BaseModel):
    """Outgoing payload for host_health events to WebUI."""

    host_id: str
    host_name: str | None
    timestamp: str
    data: dict[str, Any]
    memory: dict[str, Any] | None = None
    disk_total_gb: float | None = None
    disk_used_gb: float | None = None
    disk_available_gb: float | None = None
    memory_available_gb: float | None = None


class LogPayload(BaseModel):
    """Outgoing payload for log events to WebUI."""

    host_id: str
    host_name: str | None
    instance_id: str | None
    timestamp: str
    data: dict[str, Any]


class InstanceStatePayload(BaseModel):
    """Outgoing payload for instance_state events to WebUI."""

    host_id: str
    host_name: str | None
    instance_id: str | None
    timestamp: str
    data: dict[str, Any]

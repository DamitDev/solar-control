"""Pydantic models for deployment intents (S-040)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────


class IntentPhase(str, Enum):
    PENDING = "pending"
    RECONCILING = "reconciling"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


class ReconcileState(str, Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


VALID_PRIORITIES: frozenset[str] = frozenset({"production", "staging", "ephemeral"})
VALID_STRATEGIES: frozenset[str] = frozenset({"rolling", "immediate"})
VALID_BACKEND_TYPES: frozenset[str] = frozenset(
    {
        "llamacpp",
        "huggingface_causal",
        "huggingface_classification",
        "huggingface_embedding",
        "huggingface_vision",
    }
)
VALID_MODEL_SOURCE_SCHEMES: frozenset[str] = frozenset({"repo", "huggingface", "local"})
FORBIDDEN_BACKEND_FIELDS: frozenset[str] = frozenset(
    {
        "alias",
        "model_source",
        "host",
        "port",
        "api_key",
    }
)


# ── Request models ─────────────────────────────────────────────


class PlacementConstraints(BaseModel):
    """Placement constraints for intent (S-039 §4.5)."""

    roles: list[str] = Field(default_factory=lambda: ["inference"])
    gpu_type: str | None = None
    host_allow: list[str] = Field(default_factory=list)
    host_deny: list[str] = Field(default_factory=list)


class ResourceRequirements(BaseModel):
    """Resource hints for placement (S-039 §4.6)."""

    vram_gb: float | None = None
    ram_gb: float | None = None


class IntentCreate(BaseModel):
    """Request body for POST /api/intents (S-039 §4.1)."""

    alias: str = Field(..., min_length=1)
    model_source: str = Field(..., min_length=1)
    replicas: int = Field(default=1, ge=0)
    priority: str = Field(default="production")
    strategy: str = Field(default="rolling")
    backend: dict[str, Any] = Field(...)
    placement: PlacementConstraints = Field(default_factory=PlacementConstraints)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    metadata: dict[str, str] = Field(default_factory=dict)


# ── Response models ────────────────────────────────────────────


class ReplicaEntry(BaseModel):
    """Per-replica detail in status.replica_set (S-039 §10.1)."""

    host_id: str | None = None
    host_name: str | None = None
    instance_id: str | None = None
    state: str | None = None
    model_source: str | None = None
    healthy: bool = False
    message: str | None = None
    updated_at: str | None = None


class Condition(BaseModel):
    """Machine-readable condition (S-039 §10.3)."""

    type: str
    status: bool
    reason: str
    message: str
    last_transition: str


class StrategyProgress(BaseModel):
    """In-flight strategy progress (S-039 §11.4, extended for S-042 state machine).

    Persisted in intent status_json so strategy state survives reconciler
    restarts.  The ``phase`` field drives the strategy state machine;
    ``current_host_id`` / ``current_instance_id`` track which replacement
    is in flight; ``pending_hosts`` / ``failed_hosts`` track remaining
    and failed hosts across ticks.
    """

    strategy: str
    target_model_source: str | None = None
    phase: str | None = None
    step: str | None = None
    updated: int = 0
    in_progress: int = 0
    failed: int = 0
    current_host_id: str | None = None
    current_instance_id: str | None = None
    pending_hosts: list[str] = Field(default_factory=list)
    failed_hosts: list[str] = Field(default_factory=list)
    started_at: str | None = None
    message: str | None = None


class LastError(BaseModel):
    """Most recent reconciliation error (S-039 §10.2)."""

    code: str
    message: str
    host_id: str | None = None
    source_uri: str | None = None
    at: str


class IntentStatus(BaseModel):
    """Server-managed status object (S-039 §10.1–10.2)."""

    phase: IntentPhase = IntentPhase.PENDING
    reconcile: ReconcileState = ReconcileState.IDLE
    desired_replicas: int = 0
    observed_replicas: int = 0
    ready_replicas: int = 0
    updated_replicas: int = 0
    available: bool = False
    shortfall: int = 0
    replica_set: list[ReplicaEntry] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    strategy_progress: StrategyProgress | None = None
    last_error: LastError | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_reconciled_at: str | None = None
    ready_at: str | None = None


class IntentResponse(BaseModel):
    """Full intent record returned by GET/POST (S-039 §10.1)."""

    id: str
    alias: str
    model_source: str
    replicas: int
    priority: str
    strategy: str
    backend: dict[str, Any]
    placement: PlacementConstraints
    resources: ResourceRequirements
    metadata: dict[str, str] = Field(default_factory=dict)
    status: IntentStatus


class IntentDeletedResponse(BaseModel):
    """Response for DELETE /api/intents/{id} (S-039 §12.4)."""

    id: str
    alias: str
    phase: IntentPhase = IntentPhase.DELETING
    message: str = "Intent deletion initiated"

"""Shared host status aggregation (S-033).

Builds the ``active_jobs`` view that Solar Control publishes alongside
inference instance state, plus the ``host_status`` payload itself.

Every producer of host status must go through :func:`build_host_status_payload`.
WebUI clients replace their whole host entry when a ``host_status`` arrives, so
any emit path that omits ``active_jobs`` erases live job state on the client.
"""

import logging
from typing import Any

from app.database.jobs import job_db
from app.models.host import ActiveJobSummary, Host
from app.models.job import Job, JobStatus
from app.models.socketio import HostStatusPayload

logger = logging.getLogger(__name__)

# Statuses for which a persisted step is still actually executing.
_ACTIVE_STATUSES = (JobStatus.PENDING, JobStatus.RUNNING)

# training_config keys worth surfacing; the full config can be large.
_TRAINING_CONFIG_HINT_KEYS = ("batch_size", "max_steps", "learning_rate")


def _step_gpu_count(gpu: Any) -> int:
    """Best-effort GPU count for a translated step's ``gpu`` block.

    The block originates from the SuperNova intent and is passed through
    ``_translate_payload`` unvalidated, so it may be a ``{"count": n}`` mapping,
    a bare number, or a list of device IDs.
    """
    if not gpu:
        return 0
    if isinstance(gpu, dict):
        count = gpu.get("count", 1)
        return int(count) if isinstance(count, (int, float)) else 1
    if isinstance(gpu, (int, float)):
        return int(gpu)
    if isinstance(gpu, (list, tuple)):
        return len(gpu)
    return 1


def _build_resource_hints(
    payload: dict[str, Any], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Extract resource requirements from a translated ``JobDefinition``."""
    hints: dict[str, Any] = {}

    peak_gpu = max((_step_gpu_count(s.get("gpu")) for s in steps), default=0)
    if peak_gpu:
        hints["peak_gpu_count"] = peak_gpu

    if payload.get("min_free_disk_gb") is not None:
        hints["min_free_disk_gb"] = payload["min_free_disk_gb"]

    training_config = payload.get("training_config") or {}
    if isinstance(training_config, dict):
        subset = {
            k: training_config[k]
            for k in _TRAINING_CONFIG_HINT_KEYS
            if k in training_config
        }
        if subset:
            hints["training_config"] = subset

    return hints


def build_active_job_summary(job: Job) -> ActiveJobSummary:
    """Convert a ``Job`` record to an ``ActiveJobSummary`` for host status views.

    ``job.payload`` holds the translated host ``JobDefinition``, whose ``steps``
    is an ordered list of dicts each carrying a ``name``.
    """
    payload = job.payload or {}
    steps = [s for s in payload.get("steps", []) if isinstance(s, dict)]
    pipeline = [str(s["name"]) for s in steps if s.get("name")]

    name = payload.get("name")
    # A terminal job keeps its last step recorded, but nothing is executing.
    is_active = job.status in _ACTIVE_STATUSES

    return ActiveJobSummary(
        job_id=job.id,
        submission_id=job.submission_id,
        name=str(name) if name is not None else None,
        status=job.status.value,
        current_step_name=job.current_step_name if is_active else None,
        current_step_index=job.current_step_index if is_active else None,
        last_step_name=job.current_step_name,
        last_step_index=job.current_step_index,
        pipeline=pipeline,
        resource_hints=_build_resource_hints(payload, steps),
        started_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
    )


async def get_host_active_jobs(host_id: str) -> list[ActiveJobSummary]:
    """Aggregate active and recently-terminal jobs for *host_id*.

    Never raises: host status is emitted from the Socket.IO ``connect`` handler,
    where an exception would refuse the host's connection outright.
    """
    try:
        jobs = await job_db.get_active_by_host(host_id)
        return [build_active_job_summary(j) for j in jobs]
    except Exception:
        logger.warning(
            "Failed to aggregate active jobs for host %s", host_id, exc_info=True
        )
        return []


async def build_host_status_payload(
    host: Host, *, connected: bool
) -> HostStatusPayload:
    """Build the ``host_status`` payload for *host*, including active jobs."""
    return HostStatusPayload.from_host(
        host,
        connected=connected,
        active_jobs=await get_host_active_jobs(host.id),
    )

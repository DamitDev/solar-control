"""Aggregated resource query API (S-035).

GET /api/resources — cluster-wide view of host capacity, workloads, and reservations.
"""

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
from fastapi import APIRouter, HTTPException, Query

from app.database.hosts import host_db
from app.models import Host, HostResourceSnapshot, AggregatedResourceResponse
from app.redis_state import host_store
from app.services.host_status import get_host_active_jobs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])

_RESOURCE_TIMEOUT = 5  # seconds to wait for a host's /resources response


async def _fetch_host_resource_snapshot(
    host: Host,
) -> HostResourceSnapshot:
    """Fetch live resource data from a single solar-host.

    Proxies GET /resources from the host. On any error (connection,
    timeout, non-200), marks the host as unreachable and returns a
    degraded snapshot with DB-only data.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Base snapshot from local DB (always available)
    base = HostResourceSnapshot(
        host_id=host.id,
        host_name=host.name,
        url=host.url,
        status=host.status,
        roles=host.roles or [],
        gpu_type=host.gpu_type,
        version=host.version,
        reachable=False,
        snapshot_timestamp=now_iso,
    )

    # Try to get instances from Redis
    try:
        instances = await host_store.get_host_instances(host.id)
        base.instance_count = len(instances)
        base.running_instance_count = sum(
            1 for i in instances if i.get("status") == "running"
        )
    except Exception:
        logger.warning(
            "Failed to fetch instances from Redis for host %s",
            host.id,
            exc_info=True,
        )

    # Aggregate active job workloads from the jobs table
    base.active_jobs = await get_host_active_jobs(host.id)

    # Proxy live resource data from the host
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url.rstrip('/')}/resources"
            headers = {"X-API-Key": host.api_key}
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_RESOURCE_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    base.error = (
                        f"Host {host.name} at {host.url} returned HTTP {resp.status}"
                    )
                    return base

                data = await resp.json()
    except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError):
        base.error = f"Host unreachable at {host.url}"
        return base
    except asyncio.TimeoutError:
        base.error = f"Host timed out ({_RESOURCE_TIMEOUT}s)"
        return base
    except Exception as exc:
        base.error = f"Failed to fetch resources: {exc}"
        return base

    # Merge live resource dimensions
    base.reachable = True
    base.snapshot_timestamp = now_iso

    for dim_name in ("vram", "ram", "disk"):
        dim = data.get(dim_name)
        if dim is None:
            continue
        setattr(base, f"{dim_name}_total_gb", dim.get("total_gb"))
        setattr(base, f"{dim_name}_system_used_gb", dim.get("system_used_gb"))
        setattr(
            base, f"{dim_name}_reserved_headroom_gb", dim.get("reserved_headroom_gb")
        )
        setattr(base, f"{dim_name}_reported_used_gb", dim.get("reported_used_gb"))
        setattr(base, f"{dim_name}_available_gb", dim.get("available_gb"))

    # Merge reservation summary
    reservations = data.get("reservations", [])
    base.reservation_count = len(reservations)
    base.reservation_vram_total_gb = sum(
        float(r.get("vram_gb", 0)) for r in reservations
    )
    base.reservation_ram_total_gb = sum(float(r.get("ram_gb", 0)) for r in reservations)
    base.reservation_disk_total_gb = sum(
        float(r.get("disk_gb") or 0) for r in reservations
    )

    return base


@router.get("", response_model=AggregatedResourceResponse)
async def get_resources(
    host_id: str | None = Query(None, description="Filter to a specific host"),
    role: str | None = Query(
        None, description="Filter by host role (e.g. 'training', 'inference')"
    ),
    gpu_type: str | None = Query(
        None, description="Filter by GPU type (e.g. 'nvidia_cuda')"
    ),
    min_available_vram_gb: float | None = Query(
        None, description="Minimum available VRAM in GB"
    ),
    min_available_ram_gb: float | None = Query(
        None, description="Minimum available RAM in GB"
    ),
) -> AggregatedResourceResponse:
    """Return aggregated cluster-wide resource view.

    Fetches live resource snapshots from every known host and merges
    with locally stored metadata.  Unreachable hosts are included in
    the response with ``reachable=False`` and an error string instead
    of failing the entire request.

    The resource availability formula follows S-034 semantics:
    ``available = total - (system_used + reserved_headroom)`` where
    ``reserved_headroom = Σ max(reserved − actual, 0)`` per reservation.
    This correctly implements ``effective = max(actual, requested)`` —
    real consumption is never double-counted.
    """
    if isinstance(host_id, str):
        host = await host_db.get_host(host_id)
        if not host:
            raise HTTPException(status_code=404, detail="Host not found")
        hosts = [host]
    else:
        hosts = await host_db.get_all_hosts(
            role=role if isinstance(role, str) else None
        )

    # Fetch all host snapshots concurrently
    snapshots: list[HostResourceSnapshot] = await asyncio.gather(
        *[_fetch_host_resource_snapshot(h) for h in hosts]
    )

    # Apply response-level filters
    if isinstance(gpu_type, str):
        snapshots = [
            s
            for s in snapshots
            if s.gpu_type and s.gpu_type.lower() == gpu_type.lower()
        ]

    if isinstance(min_available_vram_gb, (int, float)):
        snapshots = [
            s
            for s in snapshots
            if s.reachable
            and s.vram_available_gb is not None
            and s.vram_available_gb >= min_available_vram_gb
        ]

    if isinstance(min_available_ram_gb, (int, float)):
        snapshots = [
            s
            for s in snapshots
            if s.reachable
            and s.ram_available_gb is not None
            and s.ram_available_gb >= min_available_ram_gb
        ]

    reachable = sum(1 for s in snapshots if s.reachable)
    unreachable = sum(1 for s in snapshots if not s.reachable)

    return AggregatedResourceResponse(
        hosts=snapshots,
        total_hosts=len(snapshots),
        reachable_hosts=reachable,
        unreachable_hosts=unreachable,
    )

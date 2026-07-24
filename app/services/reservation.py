"""Cluster-level reservation coordinator (S-038).

Orchestrates resource reservation across Solar Hosts by:
1. Querying aggregated resources (S-035)
2. Finding candidate hosts via placement policy (shared with S-041)
3. Evaluating lower-priority migration (S-037) when no immediate capacity
4. Proxying reservation calls to the selected Solar Host (S-034)
5. Tracking reservation→host mapping in Redis for release/cancel
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import aiohttp
from fastapi import HTTPException

from app.database.hosts import host_db
from app.models import Host, HostResourceSnapshot
from app.models.reservation import (
    ReservationFailure,
    ReservationReleaseResponse,
    ReservationRequest,
    ReservationResponse,
)
from app.redis_state.connection import redis_client
from app.services.placement import (
    find_candidates,
    find_displaceable_instances,
    fits_resources,
)
from app.services.migration import execute_migration

logger = logging.getLogger(__name__)

# Redis key for reservation tracking
RESERVATION_MAP = "solar:reservations"  # reservation_id → {host_id, ...}


async def _call_host_reserve(
    host: Host,
    request: ReservationRequest,
) -> dict[str, Any]:
    """Proxy a reservation request to a Solar Host (S-034 POST /resources/reservations)."""
    payload: dict[str, Any] = {
        "vram_gb": request.vram_gb,
        "ram_gb": request.ram_gb or 0.0,
        "job_id": request.job_id,
        "workload_type": request.workload_type,
        "requester": request.requester,
    }
    if request.disk_gb is not None:
        payload["disk_gb"] = request.disk_gb
    if request.ttl_seconds is not None:
        payload["ttl_seconds"] = request.ttl_seconds
    if request.expiration is not None:
        payload["expires_at"] = request.expiration

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url.rstrip('/')}/resources/reservations"
            headers = {
                "X-API-Key": host.api_key,
                "Content-Type": "application/json",
            }
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status in (200, 201):
                    return await response.json()
                text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Host '{host.name}' reservation failed: {text}",
                )
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=f"Host '{host.name}' is unreachable at {host.url}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach host '{host.name}': {e}",
        )


async def _call_host_release(
    host: Host,
    host_reservation_id: str,
) -> dict[str, Any]:
    """Proxy reservation release to Solar Host (S-034 DELETE /resources/reservations/{id})."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url.rstrip('/')}/resources/reservations/{host_reservation_id}"
            headers = {"X-API-Key": host.api_key}
            async with session.delete(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status in (200, 204):
                    return {"status": "released"}
                text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Host '{host.name}' release failed: {text}",
                )
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=f"Host '{host.name}' is unreachable at {host.url}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach host '{host.name}': {e}",
        )


async def _store_reservation(
    reservation_id: str,
    host_id: str,
    host_reservation_id: str,
    request: ReservationRequest,
) -> None:
    """Store reservation metadata in Redis for later release/cancel."""
    r = redis_client()
    data = {
        "reservation_id": reservation_id,
        "host_id": host_id,
        "host_reservation_id": host_reservation_id,
        "job_id": request.job_id,
        "requester": request.requester,
        "vram_gb": request.vram_gb,
        "ram_gb": request.ram_gb,
        "disk_gb": request.disk_gb,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await r.hset(RESERVATION_MAP, reservation_id, json.dumps(data))


async def _get_reservation(reservation_id: str) -> dict[str, Any] | None:
    """Retrieve reservation metadata from Redis."""
    r = redis_client()
    raw = await r.hget(RESERVATION_MAP, reservation_id)
    if raw is None:
        return None
    return json.loads(raw)


async def _remove_reservation(reservation_id: str) -> None:
    """Remove reservation metadata from Redis."""
    r = redis_client()
    await r.hdel(RESERVATION_MAP, reservation_id)


async def reserve_resources(
    request: ReservationRequest,
) -> ReservationResponse:
    """Execute the cluster-level reservation workflow.

    1. Fetch all hosts and resource snapshots (S-035)
    2. Find candidate hosts via placement policy (§8.4)
    3. If no candidates, evaluate migration of lower-priority workloads (S-037)
    4. Call host reservation endpoint (S-034)
    5. Return reservation details
    """
    reservation_id = str(uuid.uuid4())

    # ── 1. Fetch hosts ──────────────────────────────────────────
    hosts = await host_db.get_all_hosts()

    if not hosts:
        raise HTTPException(
            status_code=409,
            detail=ReservationFailure(
                reason="no_hosts",
                detail="No Solar Hosts are registered in the cluster",
                hosts_checked=0,
                eligible_hosts=0,
            ).model_dump(),
        )

    # ── 2. Fetch resource snapshots ─────────────────────────────
    from app.routes.management.resources import (  # noqa: F811
        _fetch_host_resource_snapshot,
    )

    snapshots_list: list[HostResourceSnapshot] = await asyncio.gather(
        *[_fetch_host_resource_snapshot(h) for h in hosts]
    )
    snapshots: dict[str, HostResourceSnapshot] = {s.host_id: s for s in snapshots_list}

    # ── 3. Find candidates ──────────────────────────────────────
    candidates = await find_candidates(
        hosts,
        snapshots,
        roles=request.host_roles,
        gpu_type=request.gpu_type,
        host_allow=request.host_allow if request.host_allow else None,
        host_deny=request.host_deny if request.host_deny else None,
        vram_gb=request.vram_gb,
        ram_gb=request.ram_gb,
        disk_gb=request.disk_gb,
        exclude_alias=request.preserve_alias,
    )

    migration_details: list[dict[str, Any]] = []
    target_host: Host | None = None
    target_snapshot: HostResourceSnapshot | None = None

    if candidates:
        target_host, target_snapshot = candidates[0]
    else:
        # ── 4. No immediate candidate — evaluate migration ──────
        logger.info(
            "No immediate candidate for reservation %s (vram=%.1f GB), "
            "evaluating migration of lower-priority workloads",
            reservation_id,
            request.vram_gb,
        )

        migration_evaluated = 0
        for host in hosts:
            snap = snapshots.get(host.id)
            if snap is None or not snap.reachable:
                continue

            displaceable = await find_displaceable_instances(
                host.id,
                request.priority,
                preserve_alias=request.preserve_alias,
            )

            if not displaceable:
                continue

            # Try migrating the lowest-priority displaceable instance
            for inst in displaceable:
                inst_id = inst.get("instance_id") or inst.get("id", "")
                inst_alias = inst.get("config", inst).get("alias") or inst.get("alias")
                if not inst_id:
                    continue

                # Find a target host for migration
                for other_host in hosts:
                    if other_host.id == host.id:
                        continue
                    other_snap = snapshots.get(other_host.id)
                    if other_snap is None or not other_snap.reachable:
                        continue

                    # Check if other host can take this instance
                    inst_vram = float(inst.get("vram_gb", 0) or 0)
                    if fits_resources(other_snap, inst_vram, None, None):
                        migration_evaluated += 1
                        try:
                            mig_result = await execute_migration(
                                instance_id=inst_id,
                                source_host_id=host.id,
                                target_host_id=other_host.id,
                                allow_production=False,
                            )
                            migration_details.append(
                                {
                                    "migration_id": mig_result.migration_id,
                                    "status": mig_result.status,
                                    "source_host_id": host.id,
                                    "target_host_id": other_host.id,
                                    "instance_id": inst_id,
                                    "alias": inst_alias,
                                }
                            )

                            if mig_result.status == "completed":
                                # Re-fetch snapshot for source host
                                snapshots[host.id] = (
                                    await _fetch_host_resource_snapshot(host)
                                )
                                # Check if this host now fits
                                if fits_resources(
                                    snapshots[host.id],
                                    request.vram_gb,
                                    request.ram_gb,
                                    request.disk_gb,
                                ):
                                    target_host = host
                                    target_snapshot = snapshots[host.id]
                                    break
                        except Exception as e:
                            logger.warning(
                                "Migration attempt failed: %s → %s: %s",
                                host.id,
                                other_host.id,
                                e,
                            )
                            continue

                if target_host is not None:
                    break

            if target_host is not None:
                break

        if target_host is None:
            raise HTTPException(
                status_code=409,
                detail=ReservationFailure(
                    reason=(
                        "insufficient_capacity"
                        if migration_evaluated == 0
                        else "migration_insufficient"
                    ),
                    detail=(
                        f"No host with sufficient capacity for "
                        f"{request.vram_gb} GB VRAM"
                        + (f" ({request.ram_gb} GB RAM)" if request.ram_gb else "")
                        + (
                            f" across {len(hosts)} hosts. "
                            f"Evaluated {migration_evaluated} migration "
                            f"candidates but none freed enough capacity."
                            if migration_evaluated > 0
                            else (
                                f" across {len(hosts)} hosts. "
                                "No lower-priority workloads available "
                                "for migration."
                            )
                        )
                    ),
                    requested=request,
                    eligible_hosts=0,
                    hosts_checked=len(hosts),
                    migration_candidates=migration_evaluated,
                ).model_dump(),
            )

    # ── 5. Call host reservation endpoint ───────────────────────
    if target_host is None or target_snapshot is None:
        raise HTTPException(
            status_code=500,
            detail="Internal error: no target host after placement",
        )

    host_result = await _call_host_reserve(target_host, request)
    host_reservation_id = host_result.get("reservation_id") or host_result.get("id", "")

    # ── 6. Store tracking metadata ──────────────────────────────
    await _store_reservation(
        reservation_id, target_host.id, host_reservation_id, request
    )

    return ReservationResponse(
        reservation_id=reservation_id,
        host_reservation_id=host_reservation_id,
        host_id=target_host.id,
        host_name=target_host.name,
        host_url=target_host.url,
        vram_gb=request.vram_gb,
        ram_gb=request.ram_gb,
        disk_gb=request.disk_gb,
        workload_type=request.workload_type,
        priority=request.priority,
        expiration=host_result.get("expires_at"),
        migrated=bool(migration_details),
        migrations=migration_details,
    )


async def release_reservation(
    reservation_id: str,
) -> ReservationReleaseResponse:
    """Release a reservation by ID, proxying to the host.

    Looks up the reservation tracking data in Redis, then calls
    DELETE /resources/reservations/{id} on the target host.
    """
    reservation = await _get_reservation(reservation_id)
    if not reservation:
        raise HTTPException(
            status_code=404,
            detail=f"Reservation '{reservation_id}' not found",
        )

    host_id = reservation["host_id"]
    host = await host_db.get_host(host_id)
    if not host:
        raise HTTPException(
            status_code=404,
            detail=(f"Host '{host_id}' not found for reservation '{reservation_id}'"),
        )

    host_reservation_id = reservation["host_reservation_id"]

    try:
        await _call_host_release(host, host_reservation_id)
        await _remove_reservation(reservation_id)

        return ReservationReleaseResponse(
            reservation_id=reservation_id,
            host_reservation_id=host_reservation_id,
            host_id=host_id,
            released=True,
            message=(f"Reservation '{reservation_id}' released on host '{host.name}'"),
        )
    except HTTPException:
        # Still remove from tracking even if host release failed
        # (the host will expire it via TTL)
        await _remove_reservation(reservation_id)
        raise

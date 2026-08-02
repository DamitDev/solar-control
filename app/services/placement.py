"""Shared placement policy helper (S-038 / S-041).

Implements the placement algorithm from docs/specs/deployment-intent.md §8.4.
Both the reservation coordinator and the intent reconciler use this module
so there is ONE placement policy, not two.
"""

import logging
from typing import Any

from app.models import Host, HostResourceSnapshot
from app.redis_state import host_store

logger = logging.getLogger(__name__)

# Priority ordering for displacement (deployment-intent.md §4.3)
PRIORITY_ORDER: dict[str, int] = {
    "ephemeral": 0,
    "staging": 1,
    "production": 2,
}


def _has_roles(host: Host, required_roles: list[str]) -> bool:
    """Check that *host* has all required roles."""
    host_roles = host.roles or []
    return all(r in host_roles for r in required_roles)


def fits_resources(
    snapshot: HostResourceSnapshot,
    vram_gb: float,
    ram_gb: float | None,
    disk_gb: float | None,
) -> bool:
    """Check if *snapshot* has sufficient available resources.

    Uses available = total - Σeffective semantics from S-034/S-035.
    """
    if not snapshot.reachable:
        return False

    if snapshot.vram_available_gb is not None:
        if snapshot.vram_available_gb < vram_gb:
            return False

    if ram_gb is not None and snapshot.ram_available_gb is not None:
        if snapshot.ram_available_gb < ram_gb:
            return False

    if disk_gb is not None and snapshot.disk_available_gb is not None:
        if snapshot.disk_available_gb < disk_gb:
            return False

    return True


async def find_candidates(
    hosts: list[Host],
    snapshots: dict[str, HostResourceSnapshot],
    *,
    roles: list[str],
    gpu_type: str | None = None,
    host_allow: list[str] | None = None,
    host_deny: list[str] | None = None,
    vram_gb: float,
    ram_gb: float | None = None,
    disk_gb: float | None = None,
    exclude_alias: str | None = None,
) -> list[tuple[Host, HostResourceSnapshot]]:
    """Find candidate hosts matching placement constraints.

    Returns candidates ranked by: most free VRAM → most free disk →
    fewest instances → host id. The first ``(host, snapshot)`` pair is
    the best choice.

    Implements deployment-intent.md §8.4 placement policy.
    """
    host_allow_set = set(host_allow) if host_allow else None
    host_deny_set = set(host_deny) if host_deny else None

    candidates: list[tuple[Host, HostResourceSnapshot]] = []

    for host in hosts:
        # Role filter
        if not _has_roles(host, roles):
            continue

        # GPU type filter
        if gpu_type is not None and host.gpu_type != gpu_type:
            continue

        # Allow/deny lists
        if host_allow_set is not None and host.id not in host_allow_set:
            continue
        if host_deny_set is not None and host.id in host_deny_set:
            continue

        # Need a resource snapshot
        snap = snapshots.get(host.id)
        if snap is None:
            continue

        # Resource fit
        if not fits_resources(snap, vram_gb, ram_gb, disk_gb):
            continue

        # One-replica-per-host check (if alias provided)
        if exclude_alias is not None:
            instances = await host_store.get_host_instances(host.id)
            conflict = any(
                (i.get("config", i).get("alias") == exclude_alias) for i in instances
            )
            if conflict:
                continue

        candidates.append((host, snap))

    # Rank: most free VRAM → most free disk → fewest running instances
    # → host id (stable tiebreak) (§8.4)
    candidates.sort(
        key=lambda pair: (
            -(pair[1].vram_available_gb or 0),
            -(pair[1].disk_available_gb or 0),
            pair[1].running_instance_count,
            pair[0].id,
        )
    )

    return candidates


def can_displace(
    candidate_priority: str,
    existing_priority: str,
) -> bool:
    """Check if *candidate_priority* may displace *existing_priority*.

    Displacement is allowed only toward strictly lower priority.
    production never displaced; equal priority never displaced.
    (deployment-intent.md §8.5)
    """
    candidate_order = PRIORITY_ORDER.get(candidate_priority)
    existing_order = PRIORITY_ORDER.get(existing_priority)

    if candidate_order is None or existing_order is None:
        return False

    return candidate_order > existing_order


async def find_displaceable_instances(
    host_id: str,
    request_priority: str,
    *,
    preserve_alias: str | None = None,
) -> list[dict[str, Any]]:
    """Find instances on *host_id* that could be displaced by *request_priority*.

    Returns instances eligible for migration, sorted lowest-priority first.
    Respects the one-replica-per-host rule: if *preserve_alias* is set,
    instances with that alias are only displaceable if more than one replica
    of that alias exists on this host.
    """
    instances = await host_store.get_host_instances(host_id)

    displaceable: list[dict[str, Any]] = []
    alias_counts: dict[str, int] = {}

    for inst in instances:
        cfg = inst.get("config", inst)
        alias = cfg.get("alias")
        if alias:
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

    for inst in instances:
        cfg = inst.get("config", inst)
        priority = cfg.get("priority") or inst.get("priority", "production")
        alias = cfg.get("alias")

        # Check one-replica preservation
        if preserve_alias and alias == preserve_alias:
            if alias_counts.get(alias, 0) <= 1:
                continue  # Must preserve at least one replica

        if can_displace(request_priority, priority):
            inst["_priority"] = priority
            displaceable.append(inst)

    # Sort by lowest priority first (ephemeral before staging)
    displaceable.sort(key=lambda i: PRIORITY_ORDER.get(i.get("_priority", ""), 99))

    return displaceable

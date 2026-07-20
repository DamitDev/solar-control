"""Host selection service for job step execution.

Selects an eligible Solar Host by role and available resources
using S-006 host status data (disk, online status, roles).
"""

import logging

from app.config import settings
from app.database.hosts import host_db
from app.models import Host, HostStatus

logger = logging.getLogger(__name__)


async def select_host(
    *,
    role: str = "training",
    min_disk_gb: float | None = None,
) -> Host:
    """Select an eligible host for a job step.

    Filters hosts by:
    1. Role match (e.g., ``role="training"``)
    2. Online status
    3. Sufficient available disk space

    The host with the most available disk is selected (best for
    large workspace creation).

    Parameters
    ----------
    role:
        Required host role. Defaults to ``"training"``.
    min_disk_gb:
        Minimum available disk space in GB. Falls back to
        ``settings.job_min_disk_gb`` if not provided.

    Returns
    -------
    Host
        The selected host.

    Raises
    ------
    RuntimeError
        If no eligible host is found.
    """
    threshold = min_disk_gb if min_disk_gb is not None else settings.job_min_disk_gb

    hosts = await host_db.get_all_hosts(role=role)
    logger.debug("select_host: found %d hosts with role=%s", len(hosts), role)

    eligible = [
        h
        for h in hosts
        if h.status == HostStatus.ONLINE
        and (h.disk_available_gb is None or h.disk_available_gb >= threshold)
    ]

    if not eligible:
        available = [
            {
                "id": h.id,
                "name": h.name,
                "status": h.status.value,
                "disk_available_gb": h.disk_available_gb,
            }
            for h in hosts
        ]
        logger.warning(
            "No eligible host for role=%s, min_disk_gb=%s. Available hosts: %s",
            role,
            threshold,
            available,
        )
        raise RuntimeError(
            f"No {role}-capable host available with " f"≥{threshold} GB free disk space"
        )

    # Pick the host with the most available disk
    eligible.sort(
        key=lambda h: h.disk_available_gb or 0.0,
        reverse=True,
    )
    selected = eligible[0]

    logger.info(
        "Selected host '%s' (%s) for job — " "disk_available_gb=%s, role=%s",
        selected.name,
        selected.id,
        selected.disk_available_gb,
        role,
    )
    return selected

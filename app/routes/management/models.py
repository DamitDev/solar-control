"""Model management API routes (under /api/models)."""

import asyncio
import logging
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database.hosts import host_db
from app.models import Host
from app.model_resolvers.parser import parse, HuggingFaceURI, RepoURI, LocalURI

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class DistributeRequest(BaseModel):
    """Request to distribute a model to a target host."""

    target_host_id: str
    source_uri: str | list[str]


class DistributeResult(BaseModel):
    """Result of a model distribution operation."""

    source_uri: str
    target_host_id: str
    target_host_name: str
    path: str
    cached: bool


async def _check_disk_space(host: Host) -> float | None:
    """
    Best-effort check of available disk space on the target host.
    Returns available GB, or None if the check fails.
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url.rstrip('/')}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Expecting structure like {"disk": {"available_gb": 10.5, ...}, ...}
                    return data.get("disk", {}).get("available_gb")
    except Exception as e:
        logger.warning("Failed to check disk space on host %s: %s", host.id, e)
    return None


async def _pull_on_host(parsed: Any, source_uri: str, host: Host) -> tuple[str, bool]:
    """
    Tells the target host to pull the model.
    Returns (local_path, cached_bool).
    """
    if isinstance(parsed, LocalURI):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot distribute local:// URIs: {source_uri}",
        )
    if isinstance(parsed, RepoURI):
        # Spec says repo:// is handled by Data Repository which is Phase 1
        raise HTTPException(
            status_code=501,
            detail=(
                f"repo:// distribution requires Data Repository integration (Phase 1). "
                f"URI: {source_uri}"
            ),
        )

    if not isinstance(parsed, HuggingFaceURI):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported URI type for distribution: {type(parsed).__name__}",
        )

    url = f"{host.url.rstrip('/')}/models/pull"
    headers = {"X-API-Key": host.api_key, "Content-Type": "application/json"}
    payload = {
        "source": "huggingface",
        "model_id": parsed.model_id,
        "source_uri": source_uri,
    }

    try:
        # Long timeout for model pull as it might involve downloading GBs
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    path = data.get("path")
                    cached = data.get("cached", False)
                    if not path:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Host '{host.name}' ({host.url}) returned success but no path for model pull.",
                        )
                    return path, cached

                # Propagate specific error codes, wrap others in 502
                PROPAGATED_CODES = {404, 507}
                try:
                    err = await response.json()
                    detail = (
                        err.get("detail") or err.get("error") or await response.text()
                    )
                except Exception:
                    detail = await response.text()

                out_code = (
                    response.status if response.status in PROPAGATED_CODES else 502
                )
                raise HTTPException(
                    status_code=out_code,
                    detail=f"Model pull failed on host '{host.name}' [{response.status}]: {detail}",
                )
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Host '{host.name}' ({host.url}) is unreachable during model pull: {e}",
        )
    except Exception as e:
        logger.exception(
            "Unexpected error during model distribution to host %s", host.id
        )
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected error during model distribution to host '{host.name}': {e}",
        )


@router.post("/distribute", response_model=list[DistributeResult])
async def distribute_model(req: DistributeRequest) -> list[DistributeResult]:
    """
    Distribute one or more models to a target host.
    """
    uris = [req.source_uri] if isinstance(req.source_uri, str) else req.source_uri
    host = await host_db.get_host(req.target_host_id)
    if not host:
        raise HTTPException(
            status_code=404, detail=f"Host '{req.target_host_id}' not found"
        )

    # Optional disk check (best-effort)
    available = await _check_disk_space(host)
    DISK_WARN_THRESHOLD_GB = 5.0
    if available is not None and available < DISK_WARN_THRESHOLD_GB:
        raise HTTPException(
            status_code=507,
            detail=f"Insufficient disk on target host '{host.name}': {available:.2f} GB available",
        )

    results = []
    for uri in uris:
        parsed = parse(uri)  # Raises 400 on bad URI
        path, cached = await _pull_on_host(parsed, uri, host)
        results.append(
            DistributeResult(
                source_uri=uri,
                target_host_id=host.id,
                target_host_name=host.name,
                path=path,
                cached=cached,
            )
        )
    return results

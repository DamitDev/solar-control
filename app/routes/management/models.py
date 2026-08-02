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
from app.model_resolvers.repo import (
    build_harbor_pull_payload,
    resolve_from_data_repository,
    validate_resolved_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Error codes that should be propagated instead of wrapped in 502
PROPAGATED_CODES = {404, 507}

# Structured error format returned by hosts for Phase 1 features
# {"error": str, "detail": str, "source_uri": str, "status_code": int}

# Error constants for distribution failures
ERR_NOT_IMPLEMENTED = "not_implemented"
ERR_NOT_FOUND = "not_found"


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


class HostModelInfo(BaseModel):
    """Info about a model on a specific host."""

    host_id: str
    host_name: str
    size_bytes: int
    path: str


class ModelAvailabilityResponse(BaseModel):
    """Aggregate view of model availability across hosts."""

    models: dict[str, list[HostModelInfo]]


@router.get("/availability", response_model=ModelAvailabilityResponse)
async def get_model_availability(
    model_name: str | None = None,
) -> ModelAvailabilityResponse:
    """
    Returns an aggregated view of model availability across all hosts.
    Optional model_name filter returns only hosts that have that specific model.
    """
    hosts = await host_db.get_all_hosts()
    results = await asyncio.gather(*[_fetch_host_models(h) for h in hosts])

    # model_name -> list[HostModelInfo]
    availability: dict[str, list[HostModelInfo]] = {}

    for host, host_models in zip(hosts, results):
        for m in host_models:
            name = m.get("name")
            if not name:
                continue

            if model_name and name != model_name:
                continue

            if name not in availability:
                availability[name] = []

            availability[name].append(
                HostModelInfo(
                    host_id=host.id,
                    host_name=host.name,
                    size_bytes=m.get("size_bytes", 0),
                    path=m.get("path", ""),
                )
            )

    return ModelAvailabilityResponse(models=availability)


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


class _StructuredPullError:
    """Internal container for structured pull failures (not raised, returned in result)."""

    def __init__(self, error: str, detail: str, source_uri: str, status_code: int):
        self.error = error
        self.detail = detail
        self.source_uri = source_uri
        self.status_code = status_code


# Long timeout: a full model pull on the host may take minutes.
_HOST_PULL_TIMEOUT_S = 300


async def _post_pull_to_host(
    host: Host, source_uri: str, payload: dict
) -> tuple[str, bool] | _StructuredPullError:
    """POST a pre-built pull payload to ``host`` and translate the response.

    Unlike ``app.model_resolvers.repo.post_harbor_pull`` (used by the
    dispatcher path, which raises HTTPException for every failure), this
    variant preserves structured per-item failures so the /distribute route
    can keep producing partial-batch results. Transport-level failures still
    raise HTTPException because they affect the whole batch, not one item.
    """
    url = f"{host.url.rstrip('/')}/models/pull"
    headers = {"X-API-Key": host.api_key, "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_HOST_PULL_TIMEOUT_S),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    path = data.get("path")
                    cached = data.get("cached", False)
                    if not path:
                        return _StructuredPullError(
                            error="bad_response",
                            detail=(
                                f"Host '{host.name}' ({host.url}) returned success "
                                "but no path"
                            ),
                            source_uri=source_uri,
                            status_code=502,
                        )
                    return path, cached

                try:
                    err = await response.json()
                    if err.get("error") and err.get("status_code"):
                        return _StructuredPullError(
                            error=err["error"],
                            detail=err.get("detail", await response.text()),
                            source_uri=err.get("source_uri", source_uri),
                            status_code=err.get("status_code", response.status),
                        )
                    detail = (
                        err.get("detail") or err.get("error") or await response.text()
                    )
                except Exception:
                    detail = await response.text()

                out_code = (
                    response.status if response.status in PROPAGATED_CODES else 502
                )
                return _StructuredPullError(
                    error="pull_failed",
                    detail=(
                        f"Model pull failed on host '{host.name}' "
                        f"[{response.status}]: {detail}"
                    ),
                    source_uri=source_uri,
                    status_code=out_code,
                )
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ) as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Host '{host.name}' ({host.url}) is unreachable during model pull: {e}"
            ),
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error during model distribution to host %s", host.id
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Unexpected error during model distribution to host "
                f"'{host.name}': {exc}"
            ),
        )


async def _pull_on_host(
    parsed: Any, source_uri: str, host: Host
) -> tuple[str, bool] | _StructuredPullError:
    """
    Tells the target host to pull the model.
    Returns (local_path, cached_bool) on success, or a _StructuredPullError on failure.

    Raises HTTPException for connection-level/dependency failures (5xx).
    Returns _StructuredPullError for expected request/content failures (4xx, 507).
    """
    if isinstance(parsed, LocalURI):
        return _StructuredPullError(
            error="validation_error",
            detail="Cannot distribute local:// URIs",
            source_uri=source_uri,
            status_code=400,
        )
    if isinstance(parsed, RepoURI):
        try:
            # The subpath (repo://name:version/subpath) is a host-side file
            # selector: Data Repository is queried with the base URI only,
            # while the full URI is forwarded to the host pull so the
            # returned path resolves to the file (D-017).
            repo_lookup_uri = source_uri
            if parsed.subpath:
                repo_lookup_uri = f"repo://{parsed.name}:{parsed.version}"
            resolved = await resolve_from_data_repository(repo_lookup_uri)
            validate_resolved_model(resolved, repo_lookup_uri)
        except HTTPException as exc:
            if exc.status_code >= 500:
                raise
            return _StructuredPullError(
                error="resolve_failed",
                detail=exc.detail,
                source_uri=source_uri,
                status_code=exc.status_code,
            )

        payload = build_harbor_pull_payload(resolved, source_uri)
        return await _post_pull_to_host(host, source_uri, payload)

    if not isinstance(parsed, HuggingFaceURI):
        return _StructuredPullError(
            error="validation_error",
            detail=f"Unsupported URI type for distribution: {type(parsed).__name__}",
            source_uri=source_uri,
            status_code=400,
        )

    payload = {
        "source": "huggingface",
        "model_id": parsed.model_id,
        "source_uri": source_uri,
    }
    return await _post_pull_to_host(host, source_uri, payload)


async def _fetch_host_models(host: Host) -> list[dict]:
    """
    Fetches the list of models from a single host.
    Returns a list of dicts with {name, size_bytes, path}, or [] on any error.
    """
    url = f"{host.url.rstrip('/')}/models"
    headers = {"X-API-Key": host.api_key}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(
                        "Host %s (%s) returned %d for GET /models",
                        host.id,
                        host.url,
                        resp.status,
                    )
    except Exception as e:
        logger.warning("Failed to fetch models from host %s: %s", host.id, e)
    return []


@router.post("/distribute", response_model=list[DistributeResult])
async def distribute_model(req: DistributeRequest) -> list[DistributeResult]:
    """
    Distribute one or more models to a target host.

    When distributing an array, each model is processed individually.
    Successful results are returned; failures are logged and skipped so
    partial results are returned for items that succeeded.
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

    results: list[DistributeResult] = []
    for uri in uris:
        try:
            parsed = parse(uri)  # Raises 400 on bad URI
        except HTTPException:
            raise

        pull_result = await _pull_on_host(parsed, uri, host)
        if isinstance(pull_result, _StructuredPullError):
            logger.warning(
                "Model pull failed on host '%s' [%d]: %s [%s]",
                host.name,
                pull_result.status_code,
                pull_result.detail,
                pull_result.source_uri,
            )
            # Skip this item and continue to next — partial results are returned
            continue
        path, cached = pull_result
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

"""Model catalog API routes (under /api/catalog) — D-018.

``GET /api/catalog/models`` proxies Data Repository's D-013 model list
(``GET /api/models``) and enriches each model with Solar runtime context:

* **deployed hosts** — hosts where the model files are present, using the
  same per-host ``GET /models`` mechanism as S-020
  (``GET /api/models/availability``). Entries are joined on the
  authoritative ``model_name`` recorded in the host manifest since D-016,
  falling back to the manifest ``name`` for legacy entries.
* **running instances** — instances currently serving the model, joined
  through their ``model_source`` (e.g. ``repo://name:version`` or
  ``huggingface://org/model``) against the host instance cache.

Data Repository remains the authority for catalog metadata and versions;
Solar Control only adds runtime context. Enrichment is best-effort: a
failed host poll never fails the whole request — it degrades
``meta.enrichment`` and the per-model ``solar.status`` derivation so the
WebUI never sees a misleading "unavailable" when the availability source
itself is down.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import aiohttp
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.database.hosts import host_db
from app.model_resolvers.parser import HuggingFaceURI, RepoURI, parse
from app.models import Host

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])


# ── Response models ───────────────────────────────────────────


class DeployedHostInfo(BaseModel):
    """A host where the model's files are present (S-020 availability)."""

    host_id: str
    host_name: str
    size_bytes: int = 0
    path: str = ""


class RunningInstanceInfo(BaseModel):
    """A running instance of the model (host instance state)."""

    host_id: str
    host_name: str
    instance_id: str


class SolarRuntimeInfo(BaseModel):
    """Solar runtime context added by Solar Control (D-018).

    ``status`` is derived per model:
    * ``available`` — at least one running instance exists.
    * ``deployed`` — on at least one host, but no instance running.
    * ``unavailable`` — not on any host and no instance running.
    * ``unknown`` — no deployment evidence and the availability source
      itself could not be reached, so absence cannot be proven.
    """

    status: Literal["available", "deployed", "unavailable", "unknown"]
    running_instances: int
    deployed_hosts: list[DeployedHostInfo] = Field(default_factory=list)
    instances: list[RunningInstanceInfo] = Field(default_factory=list)


class CatalogModelItem(BaseModel):
    """One catalog entry: Data Repository metadata + Solar runtime context."""

    name: str
    category: str
    description: str | None = None
    versions_count: int
    latest_version: str | None = None
    created_at: datetime
    solar: SolarRuntimeInfo


class CatalogMeta(BaseModel):
    """Health of the Solar-side enrichment sources for this response.

    ``ok`` — every host answered; ``partial`` — some hosts failed;
    ``unavailable`` — no host answered (enrichment is degraded, per-model
    statuses fall back to ``unknown`` unless instance evidence exists).
    """

    enrichment: Literal["ok", "partial", "unavailable"]


class CatalogResponse(BaseModel):
    total: int
    items: list[CatalogModelItem]
    meta: CatalogMeta


# ── Data Repository proxy (D-013) ─────────────────────────────


async def _list_data_repository_models(
    search: str | None, limit: int, offset: int
) -> dict[str, Any]:
    """Call Data Repository ``GET /api/models`` and return its body.

    Error mapping (mirrors ``app.model_resolvers.repo``):
      * 500 if ``DATA_REPOSITORY_URL`` is unset
      * 404/422 propagated verbatim from Data Repository
      * 502 for all other upstream errors and transport failures
    """
    if not settings.data_repository_url:
        raise HTTPException(
            status_code=500,
            detail="DATA_REPOSITORY_URL is not configured",
        )

    url = f"{settings.data_repository_url.rstrip('/')}/api/models"
    headers = {"Content-Type": "application/json"}
    if settings.data_repository_api_key:
        headers["X-API-Key"] = settings.data_repository_api_key

    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if search:
        params["search"] = search

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=settings.data_repository_timeout_s),
            ) as response:
                if response.status == 200:
                    return await response.json()

                try:
                    err = await response.json()
                    detail = err.get("detail") or err.get("error")
                except Exception:
                    detail = await response.text()

                if response.status in {404, 422}:
                    raise HTTPException(
                        status_code=response.status,
                        detail=detail or "Data Repository model list failed",
                    )

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Data Repository model list failed "
                        f"[{response.status}]: {detail}"
                    ),
                )
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Data Repository is unreachable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected error during Data Repository model list: {exc}",
        )


# ── Solar enrichment ──────────────────────────────────────────


async def _fetch_models_from_host(host: Host) -> tuple[list[dict], bool]:
    """Fetch ``GET {host.url}/models`` like S-020, reporting reachability.

    Returns ``(models, ok)``; ``ok`` is False when the host did not
    answer with 200 (unreachable, timeout, or error status).
    """
    url = f"{host.url.rstrip('/')}/models"
    headers = {"X-API-Key": host.api_key}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json(), True
                logger.warning(
                    "Host %s (%s) returned %d for GET /models",
                    host.id,
                    host.url,
                    resp.status,
                )
    except Exception as e:
        logger.warning("Failed to fetch models from host %s: %s", host.id, e)
    return [], False


async def _collect_availability() -> (
    tuple[dict[str, list[DeployedHostInfo]], Literal["ok", "partial", "unavailable"]]
):
    """Aggregate host-level model availability (same mechanism as S-020).

    Returns ``(model_name -> [DeployedHostInfo], enrichment_status)``.
    Host entries are keyed by the authoritative ``model_name`` recorded
    at pull time (D-016), falling back to the manifest ``name`` for
    legacy entries.
    """
    hosts = await host_db.get_all_hosts()
    results = await asyncio.gather(*[_fetch_models_from_host(h) for h in hosts])

    by_name: dict[str, list[DeployedHostInfo]] = {}
    failed = 0
    for host, (models, ok) in zip(hosts, results):
        if not ok:
            failed += 1
            continue
        for m in models:
            key = m.get("model_name") or m.get("name")
            if not key:
                continue
            by_name.setdefault(key, []).append(
                DeployedHostInfo(
                    host_id=host.id,
                    host_name=host.name,
                    size_bytes=m.get("size_bytes", 0),
                    path=m.get("path", ""),
                )
            )

    status: Literal["ok", "partial", "unavailable"]
    if failed == 0:
        status = "ok"
    elif failed == len(hosts) and hosts:
        status = "unavailable"
    else:
        status = "partial"
    return by_name, status


def _model_name_from_source(source_uri: str | None) -> str | None:
    """Extract the Data Repository model name from a model source URI.

    ``repo://name:version/...`` -> ``name``; ``huggingface://org/model``
    -> ``org/model``; anything unparsable -> None.
    """
    if not source_uri:
        return None
    try:
        parsed = parse(source_uri)
    except HTTPException:
        return None
    if isinstance(parsed, RepoURI):
        return parsed.name
    if isinstance(parsed, HuggingFaceURI):
        return parsed.model_id
    return None


async def _collect_running_instances() -> dict[str, list[RunningInstanceInfo]]:
    """Aggregate running instances from each host's cached instance state.

    Instances are joined to catalog models through their ``model_source``.
    The cache is best-effort: hosts that never connected contribute
    nothing, and a missing cache is indistinguishable from "no instances".
    """
    from app.socketio_app.host_handlers import get_host_instances

    hosts = await host_db.get_all_hosts()
    by_name: dict[str, list[RunningInstanceInfo]] = {}
    for host in hosts:
        instances = await get_host_instances(host.id)
        for inst in instances:
            if inst.get("status") != "running":
                continue
            config = inst.get("config") or {}
            source = config.get("model_source") or inst.get("model_source")
            name = _model_name_from_source(source)
            if not name:
                continue
            by_name.setdefault(name, []).append(
                RunningInstanceInfo(
                    host_id=host.id,
                    host_name=host.name,
                    instance_id=str(inst.get("id", "")),
                )
            )
    return by_name


def _derive_status(
    running: list[RunningInstanceInfo],
    deployed: list[DeployedHostInfo],
    availability_ok: bool,
) -> Literal["available", "deployed", "unavailable", "unknown"]:
    """Derive the WebUI-facing deployment status for a catalog model.

    Evidence-based, never misleading: running instances prove
    availability and deployed hosts prove deployment. When the
    availability source itself is down and no instance evidence exists,
    the status is ``unknown`` instead of a false ``unavailable``.
    """
    if running:
        return "available"
    if deployed:
        return "deployed"
    return "unavailable" if availability_ok else "unknown"


# ── Route ─────────────────────────────────────────────────────


@router.get("/models", response_model=CatalogResponse)
async def get_catalog_models(
    search: str | None = Query(
        None, description="Search string forwarded to Data Repository"
    ),
    limit: int = Query(50, ge=1, le=1000, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> CatalogResponse:
    """List Data Repository models enriched with Solar deployment context.

    Pagination and ``search`` are forwarded verbatim to Data Repository
    (D-013); the response keeps its ``total``/``items`` shape. Each item
    carries a ``solar`` block with the derived deployment status, running
    instance count, deployed hosts, and running instances.
    """
    repo_listing = await _list_data_repository_models(search, limit, offset)
    raw_items = repo_listing.get("items", [])
    total = repo_listing.get("total", len(raw_items))

    # Enrichment is best-effort; failures degrade metadata, never the response.
    availability, enrichment_status = await _collect_availability()
    running = await _collect_running_instances()
    availability_ok = enrichment_status == "ok"

    items: list[CatalogModelItem] = []
    for raw in raw_items:
        name = raw.get("name")
        if not name:
            continue
        deployed = availability.get(name, [])
        instances = running.get(name, [])
        items.append(
            CatalogModelItem(
                name=name,
                category=raw.get("category", "model"),
                description=raw.get("description"),
                versions_count=raw.get("versions_count", 0),
                latest_version=raw.get("latest_version"),
                created_at=raw.get("created_at"),
                solar=SolarRuntimeInfo(
                    status=_derive_status(instances, deployed, availability_ok),
                    running_instances=len(instances),
                    deployed_hosts=deployed,
                    instances=instances,
                ),
            )
        )

    return CatalogResponse(
        total=total,
        items=items,
        meta=CatalogMeta(enrichment=enrichment_status),
    )

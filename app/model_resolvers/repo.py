"""repo:// resolver: Data Repository integration + Solar Host pull (D-016).

The resolver owns the wire-level details of calling Data Repository's
``GET /api/resolve`` endpoint and forwarding the authoritative metadata to
the target host's ``POST /models/pull``. It intentionally exposes the
``build_harbor_pull_payload`` and ``post_harbor_pull`` helpers so the
``/models/distribute`` route can reuse the same payload shape without
duplicating it.
"""

import asyncio
from typing import Any

import aiohttp
from fastapi import HTTPException

from app.config import settings

# Host pull responses we propagate verbatim instead of collapsing to 502.
# 404 = model not found at source, 507 = insufficient disk on the host.
_PROPAGATED_HOST_CODES = {404, 507}

# Long timeout for the host pull leg: a full Harbor download can take minutes.
_HOST_PULL_TIMEOUT_S = 300


def to_local_uri(path: str) -> str:
    """Convert an absolute or relative host path into a ``local://`` URI."""
    if path.startswith("/"):
        return f"local:///{path.lstrip('/')}"
    return f"local://{path}"


async def resolve_from_data_repository(source_uri: str) -> dict[str, Any]:
    """Call Data Repository ``GET /api/resolve?uri=...`` and return its body.

    Raises ``HTTPException`` for any failure:
      * 500 if ``DATA_REPOSITORY_URL`` is unset
      * 404/422 propagated verbatim from Data Repository
      * 502 for all other upstream errors and transport failures
    """
    if not settings.data_repository_url:
        raise HTTPException(
            status_code=500,
            detail="DATA_REPOSITORY_URL is not configured",
        )

    url = f"{settings.data_repository_url.rstrip('/')}/api/resolve"
    headers = {"Content-Type": "application/json"}
    if settings.data_repository_api_key:
        headers["X-API-Key"] = settings.data_repository_api_key

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"uri": source_uri},
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
                        detail=detail or "Data Repository resolution failed",
                    )

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Data Repository resolution failed "
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
            detail=f"Unexpected error during Data Repository resolve: {exc}",
        )


def validate_resolved_model(payload: dict[str, Any], source_uri: str) -> None:
    """Reject artifacts that are not deployable models.

    Both ``category != "model"`` and a missing ``harbor_ref`` are treated as
    per-item client-visible problems (422). Missing ``harbor_ref`` is a
    Data Repository protocol bug, but the caller (e.g. /distribute) needs to
    be able to skip the bad item without aborting the whole batch.
    """
    category = payload.get("category")
    if category != "model":
        raise HTTPException(
            status_code=422,
            detail=(
                "Resolved artifact is not a deployable model. "
                f"Expected category 'model', got '{category}'."
            ),
        )

    harbor_ref = payload.get("harbor_ref")
    if not harbor_ref:
        raise HTTPException(
            status_code=422,
            detail=(
                "Data Repository response missing harbor_ref for "
                f"URI '{source_uri}'."
            ),
        )


def build_harbor_pull_payload(
    resolved: dict[str, Any], source_uri: str
) -> dict[str, Any]:
    """Build the ``POST /models/pull`` body for a resolved Harbor artifact.

    Centralised so the dispatcher resolver and the /distribute route stay in
    lockstep on what gets forwarded to solar-host.
    """
    payload: dict[str, Any] = {
        "source": "harbor",
        "harbor_ref": resolved["harbor_ref"],
        "source_uri": source_uri,
        "category": resolved.get("category"),
        "name": resolved.get("name"),
        "version": resolved.get("version"),
        "size_bytes": resolved.get("size_bytes"),
        "checksum": resolved.get("checksum"),
        "metadata": resolved.get("metadata"),
    }
    digest = resolved.get("checksum")
    if digest:
        payload["digest"] = digest
    return payload


async def post_harbor_pull(
    *,
    payload: dict[str, Any],
    host_url: str,
    host_api_key: str,
    host_label: str | None = None,
) -> tuple[str, bool]:
    """POST a pre-built pull payload to a host and return ``(path, cached)``.

    Raises ``HTTPException`` for transport errors and non-200 responses.
    ``host_label`` is used purely for error-message readability when callers
    have a friendlier name than the URL (e.g. a host id or display name).
    """
    label = host_label or host_url
    url = f"{host_url.rstrip('/')}/models/pull"
    headers = {"X-API-Key": host_api_key, "Content-Type": "application/json"}

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
                    if not path:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Host '{label}' returned success but no path "
                                "for model pull."
                            ),
                        )
                    return path, bool(data.get("cached", False))

                try:
                    err = await response.json()
                    detail = err.get("detail") or err.get("error")
                except Exception:
                    detail = await response.text()

                out_code = (
                    response.status
                    if response.status in _PROPAGATED_HOST_CODES
                    else 502
                )
                raise HTTPException(
                    status_code=out_code,
                    detail=(
                        f"Model pull failed on host '{label}' "
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
            detail=f"Host '{label}' is unreachable during model pull: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected error during model pull on host '{label}': {exc}",
        )


async def resolve_repo(source_uri: str, host_url: str, host_api_key: str) -> str:
    """Resolve a ``repo://`` URI end-to-end.

    Calls Data Repository for metadata, validates the result, then asks the
    target host to pull the Harbor artifact. Returns the resolved
    ``local://`` URI pointing at the host-managed model directory.
    """
    resolved = await resolve_from_data_repository(source_uri)
    validate_resolved_model(resolved, source_uri)

    payload = build_harbor_pull_payload(resolved, source_uri)
    path, _cached = await post_harbor_pull(
        payload=payload,
        host_url=host_url,
        host_api_key=host_api_key,
    )
    return to_local_uri(path)

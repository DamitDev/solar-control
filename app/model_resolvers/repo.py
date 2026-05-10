import asyncio
from typing import Any

import aiohttp
from fastapi import HTTPException

from app.config import settings
from .parser import RepoURI


async def _resolve_from_data_repository(source_uri: str) -> dict[str, Any]:
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


def _validate_resolved_model(payload: dict[str, Any], source_uri: str) -> None:
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
            status_code=502,
            detail=(
                "Data Repository response missing harbor_ref for "
                f"URI '{source_uri}'."
            ),
        )


async def _pull_from_host(
    *,
    harbor_ref: str,
    metadata: dict[str, Any],
    digest: str | None,
    source_uri: str,
    host_url: str,
    host_api_key: str,
) -> str:
    url = f"{host_url.rstrip('/')}/models/pull"
    headers = {"X-API-Key": host_api_key, "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "source": "harbor",
        "harbor_ref": harbor_ref,
        "source_uri": source_uri,
        "category": metadata.get("category"),
        "name": metadata.get("name"),
        "version": metadata.get("version"),
        "size_bytes": metadata.get("size_bytes"),
        "checksum": metadata.get("checksum"),
        "metadata": metadata.get("metadata"),
    }
    if digest:
        payload["digest"] = digest

    try:
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
                    if not path:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Host '{host_url}' returned success but no path "
                                f"for model pull."
                            ),
                        )
                    return f"local://{path}"

                try:
                    err = await response.json()
                    detail = err.get("detail") or err.get("error")
                except Exception:
                    detail = await response.text()

                out_code = response.status if response.status in {404, 507} else 502
                raise HTTPException(
                    status_code=out_code,
                    detail=(
                        f"Model pull failed on host '{host_url}' [{response.status}]: "
                        f"{detail}"
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
            detail=f"Host '{host_url}' is unreachable during model pull: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected error during model pull on host: {exc}",
        )


async def resolve_repo(
    uri: RepoURI, source_uri: str, host_url: str, host_api_key: str
) -> str:
    """
    Stub for repo:// resolver.

    Expected behavior in Phase 1 (D-016):
    1. Call Data Repository GET /api/resolve?uri={source_uri} to obtain a harbor_ref.
    2. Pull the OCI artifact from Harbor using ORAS (harbor-oci-client) into the
       host's managed models directory.
    3. Return the resolved local:// path.
    """
    # Fetch authoritative metadata from Data Repository
    resolved = await _resolve_from_data_repository(source_uri)
    _validate_resolved_model(resolved, source_uri)

    return await _pull_from_host(
        harbor_ref=resolved["harbor_ref"],
        metadata=resolved,
        digest=resolved.get("checksum"),
        source_uri=source_uri,
        host_url=host_url,
        host_api_key=host_api_key,
    )

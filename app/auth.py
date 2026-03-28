"""Authentication middleware for multi-tenant API.

Two authentication modes:
- /v1/* routes: API key looked up in api_endpoints table -> returns endpoint_id
- /api/* routes: compared against MANAGEMENT_API_KEY env var
"""

import json
import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.endpoints import endpoint_db, ApiEndpoint

logger = logging.getLogger(__name__)

ENDPOINT_CACHE_PREFIX = "solar:endpoint_cache:"
ENDPOINT_CACHE_TTL = 300  # 5 minutes


def _extract_api_key(request: Request) -> str | None:
    key = request.headers.get("X-API-Key")
    if key:
        return key
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:]
    return None


_UNAUTHORIZED_RESPONSE = {
    "error": {
        "message": "Incorrect API key provided.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}


async def invalidate_endpoint_cache() -> None:
    """Clear all cached endpoint lookups across all replicas."""
    try:
        from app.redis_state.connection import redis_client

        r = redis_client()
        keys: list[bytes] = []
        async for key in r.scan_iter(f"{ENDPOINT_CACHE_PREFIX}*"):
            keys.append(key)
        if keys:
            await r.delete(*keys)
    except Exception as e:
        logger.warning("Failed to invalidate endpoint cache: %s", e)


async def _resolve_endpoint(api_key: str) -> ApiEndpoint | None:
    try:
        from app.redis_state.connection import redis_client

        r = redis_client()
        cached = await r.get(f"{ENDPOINT_CACHE_PREFIX}{api_key}")
        if cached:
            data = json.loads(cached)
            return ApiEndpoint(**data)
    except Exception:
        pass

    ep = await endpoint_db.get_endpoint_by_api_key(api_key)
    if ep:
        try:
            from app.redis_state.connection import redis_client

            r = redis_client()
            await r.set(
                f"{ENDPOINT_CACHE_PREFIX}{api_key}",
                ep.model_dump_json(),
                ex=ENDPOINT_CACHE_TTL,
            )
        except Exception:
            pass
    return ep


async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Unified authentication middleware."""
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    public_paths = ["/health", "/ready", "/", "/docs", "/redoc", "/openapi.json"]
    if path in public_paths:
        return await call_next(request)

    if path.startswith("/socket.io"):
        return await call_next(request)

    api_key = _extract_api_key(request)
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_UNAUTHORIZED_RESPONSE,
            headers=_CORS_HEADERS,
        )

    if path.startswith("/v1/"):
        endpoint = await _resolve_endpoint(api_key)
        if endpoint:
            request.state.endpoint_id = endpoint.id
            request.state.endpoint_name = endpoint.name
            return await call_next(request)
        if api_key == settings.management_api_key:
            request.state.endpoint_id = None
            request.state.endpoint_name = None
            return await call_next(request)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_UNAUTHORIZED_RESPONSE,
            headers=_CORS_HEADERS,
        )

    if path.startswith("/api/"):
        if api_key != settings.management_api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=_UNAUTHORIZED_RESPONSE,
                headers=_CORS_HEADERS,
            )
        request.state.endpoint_id = None
        return await call_next(request)

    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=_UNAUTHORIZED_RESPONSE,
        headers=_CORS_HEADERS,
    )

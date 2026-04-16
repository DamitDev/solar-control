"""Gateway monitoring REST API endpoints (under /api/gateway)."""

from typing import Any
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Query

from app.database.logs import gateway_logger

router = APIRouter(prefix="/gateway", tags=["gateway"])


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except Exception:
        return None


@router.get("/stats")
async def get_stats(
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    request_type: str | None = Query(None),
    endpoint_id: str | None = Query(None),
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or datetime(
        now.year, now.month, now.day, tzinfo=timezone.utc
    )
    end = _parse_iso(to_ts) or now

    stats = await gateway_logger.read_stats(
        start,
        end,
        request_type=request_type if request_type and request_type != "all" else None,
        endpoint_id=endpoint_id,
    )

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        **stats,
    }


@router.get("/requests")
async def list_requests(
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    status: str = Query("all", pattern="^(all|success|error|missed)$"),
    request_type: str | None = Query(None),
    model: str | None = None,
    host_id: str | None = None,
    endpoint_id: str | None = None,
    page: int = 1,
    limit: int = 200,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or (now - timedelta(days=1))
    end = _parse_iso(to_ts) or now

    safe_limit = max(1, min(limit, 1000))
    offset = max(0, (page - 1) * safe_limit)

    items, total = await gateway_logger.read_requests_page(
        start,
        end,
        status=status if status != "all" else None,
        request_type=request_type if request_type and request_type != "all" else None,
        model=model,
        host_id=host_id,
        endpoint_id=endpoint_id,
        limit=safe_limit,
        offset=offset,
    )

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "page": page,
        "limit": safe_limit,
        "total": total,
        "items": items,
    }


@router.get("/events/recent")
async def recent_events(
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    types: str = "request_error,request_reroute",
    endpoint_id: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or (now - timedelta(days=1))
    end = _parse_iso(to_ts) or now
    wanted = [t.strip() for t in types.split(",") if t.strip()]

    safe_limit = max(1, min(limit, 5000))

    events = await gateway_logger.read_events(
        start,
        end,
        types=wanted,
        endpoint_id=endpoint_id,
        limit=safe_limit,
    )
    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "types": wanted,
        "items": events,
    }

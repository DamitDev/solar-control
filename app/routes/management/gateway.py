"""Gateway monitoring REST API endpoints (under /api/gateway)."""

from fastapi import APIRouter, Query
from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta

from app.database.logs import gateway_logger

router = APIRouter(prefix="/gateway", tags=["gateway"])


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
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
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    request_type: Optional[str] = Query(None),
    endpoint_id: Optional[str] = Query(None),
):
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or datetime(
        now.year, now.month, now.day, tzinfo=timezone.utc
    )
    end = _parse_iso(to_ts) or now

    summaries = await gateway_logger.read_requests(
        start,
        end,
        request_type=request_type if request_type and request_type != "all" else None,
        endpoint_id=endpoint_id,
    )

    completed = sum(1 for s in summaries if s.get("status") == "success")
    missed = sum(1 for s in summaries if s.get("status") == "missed")
    error = sum(1 for s in summaries if s.get("status") == "error")

    events = await gateway_logger.read_events(
        start, end, types=["request_reroute"], endpoint_id=endpoint_id
    )
    rerouted_unique = len(
        {
            e.get("data", {}).get("request_id")
            for e in events
            if e.get("data", {}).get("request_id")
        }
    )

    succ = [s for s in summaries if s.get("status") == "success"]
    p_vals = [
        int(s["prompt_tokens"])
        for s in succ
        if isinstance(s.get("prompt_tokens"), (int, float))
    ]
    c_vals = [
        int(s["completion_tokens"])
        for s in succ
        if isinstance(s.get("completion_tokens"), (int, float))
    ]
    token_in_total = sum(p_vals) if p_vals else 0
    token_out_total = sum(c_vals) if c_vals else 0

    by_model: Dict[str, Dict[str, Any]] = {}
    for s in succ:
        key = s.get("resolved_model") or s.get("model") or "unknown"
        rec = by_model.setdefault(
            key,
            {
                "model": key,
                "completed": 0,
                "token_in": 0,
                "token_out": 0,
                "dur_sum": 0.0,
            },
        )
        rec["completed"] += 1
        if isinstance(s.get("prompt_tokens"), (int, float)):
            rec["token_in"] += int(s["prompt_tokens"])
        if isinstance(s.get("completion_tokens"), (int, float)):
            rec["token_out"] += int(s["completion_tokens"])
        if isinstance(s.get("duration_s"), (int, float)):
            rec["dur_sum"] += float(s["duration_s"])

    model_rows = [
        {**v, "avg_duration_s": v["dur_sum"] / v["completed"] if v["completed"] else 0}
        for v in by_model.values()
    ]
    for r in model_rows:
        r.pop("dur_sum", None)

    by_host: Dict[str, Dict[str, Any]] = {}
    for s in summaries:
        hid = s.get("host_id")
        if not hid:
            continue
        rec = by_host.setdefault(
            hid,
            {
                "host_id": hid,
                "host_name": s.get("host_name") or hid,
                "completed": 0,
                "token_in": 0,
                "token_out": 0,
                "dur_sum": 0.0,
            },
        )
        if not rec["host_name"] or rec["host_name"] == hid:
            rec["host_name"] = s.get("host_name") or hid
        rec["completed"] += 1
        if isinstance(s.get("prompt_tokens"), (int, float)):
            rec["token_in"] += int(s["prompt_tokens"])
        if isinstance(s.get("completion_tokens"), (int, float)):
            rec["token_out"] += int(s["completion_tokens"])
        if isinstance(s.get("duration_s"), (int, float)):
            rec["dur_sum"] += float(s["duration_s"])

    host_rows = [
        {**v, "avg_duration_s": v["dur_sum"] / v["completed"] if v["completed"] else 0}
        for v in by_host.values()
    ]
    for r in host_rows:
        r.pop("dur_sum", None)

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "completed": completed,
        "missed": missed,
        "error": error,
        "rerouted_requests": rerouted_unique,
        "token_in_total": token_in_total,
        "token_out_total": token_out_total,
        "avg_tokens_in": (token_in_total / len(p_vals)) if p_vals else 0,
        "avg_tokens_out": (token_out_total / len(c_vals)) if c_vals else 0,
        "models": model_rows,
        "hosts": host_rows,
    }


@router.get("/requests")
async def list_requests(
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    status: str = Query("all", pattern="^(all|success|error|missed)$"),
    request_type: Optional[str] = Query(None),
    model: Optional[str] = None,
    host_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    page: int = 1,
    limit: int = 200,
):
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or (now - timedelta(days=1))
    end = _parse_iso(to_ts) or now

    items = await gateway_logger.read_requests(
        start,
        end,
        status=status if status != "all" else None,
        request_type=request_type if request_type and request_type != "all" else None,
        model=model,
        host_id=host_id,
        endpoint_id=endpoint_id,
    )

    total = len(items)
    start_idx = max(0, (page - 1) * max(1, limit))
    page_items = items[start_idx : start_idx + max(1, limit)]

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "page": page,
        "limit": limit,
        "total": total,
        "items": page_items,
    }


@router.get("/events/recent")
async def recent_events(
    from_ts: Optional[str] = Query(None, alias="from"),
    to_ts: Optional[str] = Query(None, alias="to"),
    types: str = "request_error,request_reroute",
    endpoint_id: Optional[str] = None,
    limit: int = 1000,
):
    now = datetime.now(timezone.utc)
    start = _parse_iso(from_ts) or (now - timedelta(days=1))
    end = _parse_iso(to_ts) or now
    wanted = [t.strip() for t in types.split(",") if t.strip()]

    events = await gateway_logger.read_events(
        start, end, types=wanted, endpoint_id=endpoint_id
    )
    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "types": wanted,
        "items": events[-limit:] if len(events) > limit else events,
    }

"""Gateway event logging - async PostgreSQL storage.

Stores events and request summaries in PostgreSQL tables:
- gateway_events  -- all raw events
- gateway_requests -- request summaries (on completion)

Uses a write queue with periodic batch inserts for high throughput.
Now uses the shared connection pool and tags requests with endpoint_id.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .connection import db_pool

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def classify_request_type(endpoint: Optional[str]) -> str:
    if not endpoint:
        return "unknown"
    ep = endpoint.lower()
    if "/embeddings" in ep:
        return "embedding"
    if "/chat/completions" in ep:
        return "chat"
    if "/completions" in ep:
        return "completion"
    if "/classify" in ep:
        return "classification"
    if "/rerank" in ep:
        return "rerank"
    if "/tokenize" in ep:
        return "tokenize"
    if "/detokenize" in ep:
        return "detokenize"
    return "unknown"


@dataclass
class RequestInProgress:
    request_id: str
    request_type: str = "unknown"
    model: Optional[str] = None
    resolved_model: Optional[str] = None
    endpoint: Optional[str] = None
    endpoint_id: Optional[str] = None
    client_ip: Optional[str] = None
    stream: Optional[bool] = None
    start_timestamp: Optional[str] = None
    host_id: Optional[str] = None
    host_name: Optional[str] = None
    instance_id: Optional[str] = None
    instance_url: Optional[str] = None
    attempts: int = 0


@dataclass
class RequestSummary:
    request_id: str
    request_type: str
    status: str
    model: Optional[str]
    resolved_model: Optional[str]
    endpoint: Optional[str]
    endpoint_id: Optional[str]
    client_ip: Optional[str]
    stream: Optional[bool]
    attempts: int
    start_timestamp: Optional[str]
    end_timestamp: str
    duration_s: Optional[float]
    host_id: Optional[str]
    host_name: Optional[str]
    instance_id: Optional[str]
    instance_url: Optional[str]
    error_message: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None


class GatewayLogger:
    """Async gateway event logger with PostgreSQL storage."""

    FLUSH_INTERVAL_S = 1.0
    MAX_BUFFER_SIZE = 100

    def __init__(self) -> None:
        self._inflight: Dict[str, RequestInProgress] = {}
        self._lock = asyncio.Lock()
        self._event_buffer: List[dict] = []
        self._request_buffer: List[dict] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        self._stop_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._flush_task:
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self._flush_all()

    async def _flush_loop(self) -> None:
        while self._stop_event and not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.FLUSH_INTERVAL_S
                )
            except asyncio.TimeoutError:
                pass
            await self._flush_all()

    async def _flush_all(self) -> None:
        async with self._buffer_lock:
            events = list(self._event_buffer)
            requests = list(self._request_buffer)
            self._event_buffer.clear()
            self._request_buffer.clear()

        try:
            pool = db_pool()
        except RuntimeError:
            return

        if events:
            try:
                async with pool.acquire() as conn:
                    await conn.executemany(
                        """INSERT INTO gateway_events (event_type, request_id, endpoint_id, data, timestamp)
                           VALUES ($1, $2, $3::uuid, $4::jsonb, $5)""",
                        [
                            (
                                e["event_type"],
                                e.get("request_id"),
                                (
                                    uuid.UUID(e["endpoint_id"])
                                    if e.get("endpoint_id")
                                    else None
                                ),
                                json.dumps(e["data"], default=str),
                                e["timestamp"],
                            )
                            for e in events
                        ],
                    )
            except Exception as exc:
                logger.error("Failed to flush events: %s", exc)

        if requests:
            try:
                async with pool.acquire() as conn:
                    await conn.executemany(
                        """INSERT INTO gateway_requests (
                            request_id, request_type, status, model, resolved_model,
                            endpoint, endpoint_id, client_ip, stream, attempts,
                            start_timestamp, end_timestamp, duration_s, host_id,
                            host_name, instance_id, instance_url, error_message,
                            prompt_tokens, completion_tokens, total_tokens,
                            decode_tps, decode_ms_per_token
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7::uuid,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)
                        ON CONFLICT (request_id) DO NOTHING""",
                        [
                            (
                                r["request_id"],
                                r.get("request_type"),
                                r["status"],
                                r.get("model"),
                                r.get("resolved_model"),
                                r.get("endpoint"),
                                (
                                    uuid.UUID(r["endpoint_id"])
                                    if r.get("endpoint_id")
                                    else None
                                ),
                                r.get("client_ip"),
                                r.get("stream"),
                                r.get("attempts", 1),
                                _parse_ts(r.get("start_timestamp")),
                                _parse_ts(r["end_timestamp"]),
                                r.get("duration_s"),
                                r.get("host_id"),
                                r.get("host_name"),
                                r.get("instance_id"),
                                r.get("instance_url"),
                                r.get("error_message"),
                                r.get("prompt_tokens"),
                                r.get("completion_tokens"),
                                r.get("total_tokens"),
                                r.get("decode_tps"),
                                r.get("decode_ms_per_token"),
                            )
                            for r in requests
                        ],
                    )
            except Exception as exc:
                logger.error("Failed to flush requests: %s", exc)

    async def _queue_event(
        self,
        event_type: str,
        request_id: Optional[str],
        endpoint_id: Optional[str],
        data: dict,
        timestamp: datetime,
    ) -> None:
        should_flush = False
        async with self._buffer_lock:
            self._event_buffer.append(
                {
                    "event_type": event_type,
                    "request_id": request_id,
                    "endpoint_id": endpoint_id,
                    "data": data,
                    "timestamp": timestamp,
                }
            )
            if len(self._event_buffer) >= self.MAX_BUFFER_SIZE:
                should_flush = True
        if should_flush:
            asyncio.create_task(self._flush_all())

    async def _queue_request(self, summary_dict: dict) -> None:
        async with self._buffer_lock:
            self._request_buffer.append(summary_dict)

    async def log_event(
        self, event: Dict[str, Any], *, endpoint_id: Optional[str] = None
    ) -> Optional[RequestSummary]:
        """Log a gateway event.

        Returns a RequestSummary when the event completes a request lifecycle.
        """
        etype = event.get("type")
        data = event.get("data") or {}
        timestamp = data.get("timestamp") or event.get("timestamp") or _utc_now_iso()
        request_id = data.get("request_id")

        if "timestamp" not in event:
            event["timestamp"] = timestamp

        ts_dt = _parse_ts(timestamp) or datetime.now(timezone.utc)
        await self._queue_event(
            etype or "unknown", request_id, endpoint_id, data, ts_dt
        )

        if not request_id:
            return None

        summary = None
        async with self._lock:
            if etype == "request_start":
                ep = data.get("endpoint")
                self._inflight[request_id] = RequestInProgress(
                    request_id=request_id,
                    request_type=classify_request_type(ep),
                    model=data.get("model"),
                    endpoint=ep,
                    endpoint_id=endpoint_id,
                    client_ip=data.get("client_ip"),
                    stream=(
                        bool(data.get("stream"))
                        if data.get("stream") is not None
                        else None
                    ),
                    start_timestamp=timestamp,
                )

            elif etype == "request_routed":
                rip = self._inflight.get(request_id)
                if not rip:
                    ep = data.get("endpoint")
                    rip = RequestInProgress(
                        request_id=request_id,
                        request_type=classify_request_type(ep),
                        model=data.get("model"),
                        endpoint=ep,
                        endpoint_id=endpoint_id,
                        start_timestamp=timestamp,
                    )
                    self._inflight[request_id] = rip

                rip.attempts += 1
                rip.resolved_model = data.get("resolved_model") or rip.resolved_model
                rip.host_id = data.get("host_id") or rip.host_id
                rip.host_name = data.get("host_name") or rip.host_name
                rip.instance_id = data.get("instance_id") or rip.instance_id
                rip.instance_url = data.get("instance_url") or rip.instance_url
                rip.client_ip = data.get("client_ip") or rip.client_ip

            elif etype in ("request_success", "request_error"):
                rip = self._inflight.pop(request_id, None)
                if not rip:
                    ep = data.get("endpoint")
                    rip = RequestInProgress(
                        request_id=request_id,
                        request_type=classify_request_type(ep),
                        model=data.get("model"),
                        endpoint=ep,
                        endpoint_id=endpoint_id,
                        start_timestamp=timestamp,
                    )

                status = (
                    "success"
                    if etype == "request_success"
                    else self._classify_error_status(data.get("error_message"))
                )
                duration = data.get("duration")

                p_tok = (
                    data.get("prompt_tokens")
                    if isinstance(data.get("prompt_tokens"), (int, float))
                    else None
                )
                c_tok = (
                    data.get("completion_tokens")
                    if isinstance(data.get("completion_tokens"), (int, float))
                    else None
                )
                t_tok = (
                    data.get("total_tokens")
                    if isinstance(data.get("total_tokens"), (int, float))
                    else None
                )
                if t_tok is None and p_tok is not None and c_tok is not None:
                    t_tok = int(p_tok) + int(c_tok)

                decode_tps = (
                    float(data["decode_tps"])
                    if isinstance(data.get("decode_tps"), (int, float))
                    else None
                )
                decode_ms = (
                    float(data["decode_ms_per_token"])
                    if isinstance(data.get("decode_ms_per_token"), (int, float))
                    else None
                )

                summary = RequestSummary(
                    request_id=request_id,
                    request_type=rip.request_type,
                    status=status,
                    model=rip.model,
                    resolved_model=rip.resolved_model,
                    endpoint=rip.endpoint,
                    endpoint_id=rip.endpoint_id or endpoint_id,
                    client_ip=rip.client_ip,
                    stream=rip.stream,
                    attempts=max(1, rip.attempts),
                    start_timestamp=rip.start_timestamp,
                    end_timestamp=timestamp,
                    duration_s=(
                        float(duration)
                        if duration is not None
                        else self._compute_duration(rip.start_timestamp, timestamp)
                    ),
                    host_id=rip.host_id or data.get("host_id"),
                    host_name=rip.host_name or data.get("host_name"),
                    instance_id=rip.instance_id or data.get("instance_id"),
                    instance_url=rip.instance_url,
                    error_message=data.get("error_message"),
                    prompt_tokens=int(p_tok) if p_tok is not None else None,
                    completion_tokens=int(c_tok) if c_tok is not None else None,
                    total_tokens=int(t_tok) if t_tok is not None else None,
                    decode_tps=decode_tps,
                    decode_ms_per_token=decode_ms,
                )
                await self._queue_request(asdict(summary))

        return summary

    def _compute_duration(
        self, start_iso: Optional[str], end_iso: str
    ) -> Optional[float]:
        if not start_iso:
            return None
        try:
            s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            return max(0.0, (e - s).total_seconds())
        except Exception:
            return None

    def _classify_error_status(self, message: Optional[str]) -> str:
        if not message:
            return "error"
        m = message.lower()
        if "no instances available" in m or ("model" in m and "not found" in m):
            return "missed"
        return "error"

    async def read_requests(
        self,
        start: datetime,
        end: datetime,
        *,
        status: Optional[str] = None,
        request_type: Optional[str] = None,
        model: Optional[str] = None,
        host_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> List[dict]:
        await self._flush_all()

        try:
            pool = db_pool()
        except RuntimeError:
            return []

        query = "SELECT * FROM gateway_requests WHERE end_timestamp >= $1 AND end_timestamp <= $2"
        params: list = [start, end]
        idx = 3

        if status and status != "all":
            query += f" AND status = ${idx}"
            params.append(status)
            idx += 1
        if request_type and request_type != "all":
            query += f" AND request_type = ${idx}"
            params.append(request_type)
            idx += 1
        if model:
            query += f" AND (model = ${idx} OR resolved_model = ${idx})"
            params.append(model)
            idx += 1
        if host_id:
            query += f" AND host_id = ${idx}"
            params.append(host_id)
            idx += 1
        if endpoint_id:
            query += f" AND endpoint_id = ${idx}::uuid"
            params.append(uuid.UUID(endpoint_id))
            idx += 1

        query += " ORDER BY end_timestamp DESC"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            d = dict(row)
            d.pop("id", None)
            if d.get("endpoint_id"):
                d["endpoint_id"] = str(d["endpoint_id"])
            for field in ("start_timestamp", "end_timestamp"):
                if isinstance(d.get(field), datetime):
                    d[field] = d[field].isoformat()
            results.append(d)
        return results

    async def read_events(
        self,
        start: datetime,
        end: datetime,
        *,
        types: Optional[List[str]] = None,
        endpoint_id: Optional[str] = None,
    ) -> List[dict]:
        await self._flush_all()

        try:
            pool = db_pool()
        except RuntimeError:
            return []

        query = "SELECT * FROM gateway_events WHERE timestamp >= $1 AND timestamp <= $2"
        params: list = [start, end]
        idx = 3

        if types:
            placeholders = ", ".join(f"${idx + i}" for i in range(len(types)))
            query += f" AND event_type IN ({placeholders})"
            params.extend(types)
            idx += len(types)

        if endpoint_id:
            query += f" AND endpoint_id = ${idx}::uuid"
            params.append(uuid.UUID(endpoint_id))
            idx += 1

        query += " ORDER BY timestamp ASC"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            evt = {
                "type": row["event_type"],
                "data": (
                    json.loads(row["data"])
                    if isinstance(row["data"], str)
                    else row["data"]
                ),
                "timestamp": row["timestamp"].isoformat(),
            }
            if row.get("endpoint_id"):
                evt["endpoint_id"] = str(row["endpoint_id"])
            results.append(evt)
        return results


gateway_logger = GatewayLogger()

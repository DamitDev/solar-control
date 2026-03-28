"""Gateway event logging - async PostgreSQL storage via SQLAlchemy.

Stores events and request summaries in PostgreSQL tables:
- gateway_events  -- all raw events
- gateway_requests -- request summaries (on completion)

Uses a write queue with periodic batch inserts for high throughput.
"""

import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Any
from datetime import datetime, timezone

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .connection import get_session_factory
from .tables import GatewayEventRow, GatewayRequestRow

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def classify_request_type(endpoint: str | None) -> str:
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
    model: str | None = None
    resolved_model: str | None = None
    endpoint: str | None = None
    endpoint_id: str | None = None
    client_ip: str | None = None
    stream: bool | None = None
    start_timestamp: str | None = None
    host_id: str | None = None
    host_name: str | None = None
    instance_id: str | None = None
    instance_url: str | None = None
    attempts: int = 0


@dataclass
class RequestSummary:
    request_id: str
    request_type: str
    status: str
    model: str | None
    resolved_model: str | None
    endpoint: str | None
    endpoint_id: str | None
    client_ip: str | None
    stream: bool | None
    attempts: int
    start_timestamp: str | None
    end_timestamp: str
    duration_s: float | None
    host_id: str | None
    host_name: str | None
    instance_id: str | None
    instance_url: str | None
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    decode_tps: float | None = None
    decode_ms_per_token: float | None = None


class GatewayLogger:
    """Async gateway event logger with PostgreSQL storage."""

    FLUSH_INTERVAL_S = 1.0
    MAX_BUFFER_SIZE = 100

    def __init__(self) -> None:
        self._inflight: dict[str, RequestInProgress] = {}
        self._lock = asyncio.Lock()
        self._event_buffer: list[dict[str, Any]] = []
        self._request_buffer: list[dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

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
            session_factory = get_session_factory()
        except RuntimeError:
            return

        if events:
            try:
                async with session_factory() as session:
                    for e in events:
                        session.add(
                            GatewayEventRow(
                                event_type=e["event_type"],
                                request_id=e.get("request_id"),
                                endpoint_id=e.get("endpoint_id"),
                                data=e["data"],
                                timestamp=e["timestamp"],
                            )
                        )
                    await session.commit()
            except Exception as exc:
                logger.error("Failed to flush events: %s", exc)

        if requests:
            try:
                async with session_factory() as session:
                    for r in requests:
                        stmt = (
                            pg_insert(GatewayRequestRow)
                            .values(
                                request_id=r["request_id"],
                                request_type=r.get("request_type"),
                                status=r["status"],
                                model=r.get("model"),
                                resolved_model=r.get("resolved_model"),
                                endpoint=r.get("endpoint"),
                                endpoint_id=r.get("endpoint_id"),
                                client_ip=r.get("client_ip"),
                                stream=r.get("stream"),
                                attempts=r.get("attempts", 1),
                                start_timestamp=_parse_ts(r.get("start_timestamp")),
                                end_timestamp=_parse_ts(r["end_timestamp"]),
                                duration_s=r.get("duration_s"),
                                host_id=r.get("host_id"),
                                host_name=r.get("host_name"),
                                instance_id=r.get("instance_id"),
                                instance_url=r.get("instance_url"),
                                error_message=r.get("error_message"),
                                prompt_tokens=r.get("prompt_tokens"),
                                completion_tokens=r.get("completion_tokens"),
                                total_tokens=r.get("total_tokens"),
                                decode_tps=r.get("decode_tps"),
                                decode_ms_per_token=r.get("decode_ms_per_token"),
                            )
                            .on_conflict_do_nothing(index_elements=["request_id"])
                        )
                        await session.execute(stmt)
                    await session.commit()
            except Exception as exc:
                logger.error("Failed to flush requests: %s", exc)

    async def _queue_event(
        self,
        event_type: str,
        request_id: str | None,
        endpoint_id: str | None,
        data: dict[str, Any],
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

    async def _queue_request(self, summary_dict: dict[str, Any]) -> None:
        async with self._buffer_lock:
            self._request_buffer.append(summary_dict)

    async def log_event(
        self, event: dict[str, Any], *, endpoint_id: str | None = None
    ) -> RequestSummary | None:
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

    def _compute_duration(self, start_iso: str | None, end_iso: str) -> float | None:
        if not start_iso:
            return None
        try:
            s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            return max(0.0, (e - s).total_seconds())
        except Exception:
            return None

    def _classify_error_status(self, message: str | None) -> str:
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
        status: str | None = None,
        request_type: str | None = None,
        model: str | None = None,
        host_id: str | None = None,
        endpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        await self._flush_all()

        try:
            session_factory = get_session_factory()
        except RuntimeError:
            return []

        R = GatewayRequestRow
        conditions = [R.end_timestamp >= start, R.end_timestamp <= end]

        if status and status != "all":
            conditions.append(R.status == status)
        if request_type and request_type != "all":
            conditions.append(R.request_type == request_type)
        if model:
            conditions.append((R.model == model) | (R.resolved_model == model))
        if host_id:
            conditions.append(R.host_id == host_id)
        if endpoint_id:
            conditions.append(R.endpoint_id == endpoint_id)

        stmt = select(R).where(and_(*conditions)).order_by(R.end_timestamp.desc())

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        results: list[dict[str, Any]] = []
        for row in rows:
            d: dict[str, Any] = {
                "request_id": row.request_id,
                "request_type": row.request_type,
                "status": row.status,
                "model": row.model,
                "resolved_model": row.resolved_model,
                "endpoint": row.endpoint,
                "endpoint_id": str(row.endpoint_id) if row.endpoint_id else None,
                "client_ip": row.client_ip,
                "stream": row.stream,
                "attempts": row.attempts,
                "start_timestamp": (
                    row.start_timestamp.isoformat() if row.start_timestamp else None
                ),
                "end_timestamp": (
                    row.end_timestamp.isoformat() if row.end_timestamp else None
                ),
                "duration_s": row.duration_s,
                "host_id": row.host_id,
                "host_name": row.host_name,
                "instance_id": row.instance_id,
                "instance_url": row.instance_url,
                "error_message": row.error_message,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
                "decode_tps": row.decode_tps,
                "decode_ms_per_token": row.decode_ms_per_token,
            }
            results.append(d)
        return results

    async def read_events(
        self,
        start: datetime,
        end: datetime,
        *,
        types: list[str] | None = None,
        endpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        await self._flush_all()

        try:
            session_factory = get_session_factory()
        except RuntimeError:
            return []

        E = GatewayEventRow
        conditions = [E.timestamp >= start, E.timestamp <= end]

        if types:
            conditions.append(E.event_type.in_(types))
        if endpoint_id:
            conditions.append(E.endpoint_id == endpoint_id)

        stmt = select(E).where(and_(*conditions)).order_by(E.timestamp.asc())

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        results: list[dict[str, Any]] = []
        for row in rows:
            evt: dict[str, Any] = {
                "type": row.event_type,
                "data": row.data if isinstance(row.data, dict) else {},
                "timestamp": row.timestamp.isoformat(),
            }
            if row.endpoint_id:
                evt["endpoint_id"] = str(row.endpoint_id)
            results.append(evt)
        return results


gateway_logger = GatewayLogger()

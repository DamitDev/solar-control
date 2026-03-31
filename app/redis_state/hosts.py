"""Host Socket.IO connection state stored in Redis.

Tracks which hosts are connected via Socket.IO, their instance caches,
and pending-approval hosts. All state is shared across replicas.
"""

import json
import time
from typing import Any

from .connection import redis_client

SID_MAP = "solar:hosts:sids"
CONNECTED_MAP = "solar:hosts:connected"
INSTANCES_MAP = "solar:hosts:instances"

DISCONNECT_TS_MAP = "solar:hosts:disconnect_ts"
RECONNECT_REQ_MAP = "solar:hosts:reconnect_req_ts"

PENDING_MAP = "solar:hosts:pending"
PENDING_SID_MAP = "solar:hosts:pending_sids"


class HostConnectionStore:
    """Host Socket.IO connection state in Redis."""

    async def register_host(self, sid: str, host_id: str) -> None:
        r = redis_client()
        pipe = r.pipeline()
        pipe.hset(SID_MAP, sid, host_id)
        pipe.hset(CONNECTED_MAP, host_id, sid)
        await pipe.execute()

    async def unregister_host_by_sid(self, sid: str) -> str | None:
        r = redis_client()
        host_id = await r.hget(SID_MAP, sid)
        if host_id:
            pipe = r.pipeline()
            pipe.hdel(SID_MAP, sid)
            pipe.hdel(CONNECTED_MAP, host_id)
            await pipe.execute()
        else:
            await r.hdel(SID_MAP, sid)
        return host_id

    async def get_host_id_for_sid(self, sid: str) -> str | None:
        r = redis_client()
        return await r.hget(SID_MAP, sid)

    async def is_host_connected(self, host_id: str) -> bool:
        r = redis_client()
        return await r.hexists(CONNECTED_MAP, host_id)

    async def get_connected_host_ids(self) -> list[str]:
        r = redis_client()
        return list(await r.hkeys(CONNECTED_MAP))

    async def set_host_instances(
        self, host_id: str, instances: list[dict[str, Any]]
    ) -> None:
        r = redis_client()
        await r.hset(INSTANCES_MAP, host_id, json.dumps(instances))

    async def get_host_instances(self, host_id: str) -> list[dict[str, Any]]:
        r = redis_client()
        raw = await r.hget(INSTANCES_MAP, host_id)
        if raw is None:
            return []
        return json.loads(raw)

    async def remove_host_instances(self, host_id: str) -> None:
        r = redis_client()
        await r.hdel(INSTANCES_MAP, host_id)

    # ── Disconnect timestamp tracking ─────────────────────────

    async def set_disconnect_time(self, host_id: str) -> None:
        r = redis_client()
        await r.hset(DISCONNECT_TS_MAP, host_id, str(time.time()))

    async def get_disconnect_time(self, host_id: str) -> float | None:
        r = redis_client()
        raw = await r.hget(DISCONNECT_TS_MAP, host_id)
        return float(raw) if raw else None

    async def clear_disconnect_time(self, host_id: str) -> None:
        r = redis_client()
        await r.hdel(DISCONNECT_TS_MAP, host_id)

    async def set_reconnect_request_time(self, host_id: str) -> None:
        r = redis_client()
        await r.hset(RECONNECT_REQ_MAP, host_id, str(time.time()))

    async def get_reconnect_request_time(self, host_id: str) -> float | None:
        r = redis_client()
        raw = await r.hget(RECONNECT_REQ_MAP, host_id)
        return float(raw) if raw else None

    async def clear_reconnect_request_time(self, host_id: str) -> None:
        r = redis_client()
        await r.hdel(RECONNECT_REQ_MAP, host_id)

    async def add_pending(
        self, pending_id: str, sid: str, data: dict[str, Any]
    ) -> None:
        r = redis_client()
        pipe = r.pipeline()
        pipe.hset(PENDING_MAP, pending_id, json.dumps(data))
        pipe.hset(PENDING_SID_MAP, sid, pending_id)
        await pipe.execute()

    async def get_pending(self, pending_id: str) -> dict[str, Any] | None:
        r = redis_client()
        raw = await r.hget(PENDING_MAP, pending_id)
        if raw is None:
            return None
        return json.loads(raw)

    async def get_pending_id_for_sid(self, sid: str) -> str | None:
        r = redis_client()
        return await r.hget(PENDING_SID_MAP, sid)

    async def get_all_pending(self) -> list[dict[str, Any]]:
        r = redis_client()
        raw = await r.hgetall(PENDING_MAP)
        return [json.loads(v) for v in raw.values()]

    async def remove_pending(self, pending_id: str) -> dict[str, Any] | None:
        r = redis_client()
        raw = await r.hget(PENDING_MAP, pending_id)
        if raw is None:
            return None
        data: dict[str, Any] = json.loads(raw)
        sid = data.get("sid")
        pipe = r.pipeline()
        pipe.hdel(PENDING_MAP, pending_id)
        if sid:
            pipe.hdel(PENDING_SID_MAP, sid)
        await pipe.execute()
        return data

    async def remove_pending_by_sid(self, sid: str) -> str | None:
        r = redis_client()
        pending_id = await r.hget(PENDING_SID_MAP, sid)
        if pending_id:
            pipe = r.pipeline()
            pipe.hdel(PENDING_SID_MAP, sid)
            pipe.hdel(PENDING_MAP, pending_id)
            await pipe.execute()
        return pending_id

    async def update_pending(self, pending_id: str, data: dict[str, Any]) -> None:
        r = redis_client()
        await r.hset(PENDING_MAP, pending_id, json.dumps(data))

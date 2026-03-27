"""Host Socket.IO connection state stored in Redis.

Tracks which hosts are connected via Socket.IO, their instance caches,
and pending-approval hosts. All state is shared across replicas.
"""

import json
from typing import List, Optional

from .connection import redis_client

SID_MAP = "solar:hosts:sids"  # hash: sid -> host_id
CONNECTED_MAP = "solar:hosts:connected"  # hash: host_id -> sid
INSTANCES_MAP = "solar:hosts:instances"  # hash: host_id -> json instances

PENDING_MAP = "solar:hosts:pending"  # hash: pending_id -> json data
PENDING_SID_MAP = "solar:hosts:pending_sids"  # hash: sid -> pending_id
PENDING_KEY_MAP = "solar:hosts:pending_keys"  # hash: api_key -> pending_id


class HostConnectionStore:
    """Host Socket.IO connection state in Redis."""

    # ── Active hosts ──

    async def register_host(self, sid: str, host_id: str) -> Optional[str]:
        """Register a host connection, cleaning up any stale session.

        Returns the old sid if the host was already connected under a
        different sid (caller should disconnect it), or None.
        """
        r = redis_client()
        old_sid = await r.hget(CONNECTED_MAP, host_id)

        pipe = r.pipeline()
        if old_sid and old_sid != sid:
            pipe.hdel(SID_MAP, old_sid)
        pipe.hset(SID_MAP, sid, host_id)
        pipe.hset(CONNECTED_MAP, host_id, sid)
        await pipe.execute()

        return old_sid if old_sid and old_sid != sid else None

    async def unregister_host_by_sid(self, sid: str) -> Optional[str]:
        """Remove a host connection by sid.

        Only clears CONNECTED_MAP when *this* sid is still the active one
        for the host, preventing a stale disconnect from corrupting a newer
        connection.  Returns host_id only when this was the active session
        (so the caller knows to run the offline workflow).
        """
        r = redis_client()
        host_id = await r.hget(SID_MAP, sid)
        if not host_id:
            await r.hdel(SID_MAP, sid)
            return None

        current_sid = await r.hget(CONNECTED_MAP, host_id)
        pipe = r.pipeline()
        pipe.hdel(SID_MAP, sid)
        if current_sid == sid:
            pipe.hdel(CONNECTED_MAP, host_id)
        await pipe.execute()

        return host_id if current_sid == sid else None

    async def get_host_id_for_sid(self, sid: str) -> Optional[str]:
        r = redis_client()
        return await r.hget(SID_MAP, sid)

    async def is_host_connected(self, host_id: str) -> bool:
        r = redis_client()
        return await r.hexists(CONNECTED_MAP, host_id)

    async def get_connected_host_ids(self) -> List[str]:
        r = redis_client()
        return list(await r.hkeys(CONNECTED_MAP))

    # ── Instance cache ──

    async def set_host_instances(self, host_id: str, instances: list) -> None:
        r = redis_client()
        await r.hset(INSTANCES_MAP, host_id, json.dumps(instances))

    async def get_host_instances(self, host_id: str) -> list:
        r = redis_client()
        raw = await r.hget(INSTANCES_MAP, host_id)
        if raw is None:
            return []
        return json.loads(raw)

    async def remove_host_instances(self, host_id: str) -> None:
        r = redis_client()
        await r.hdel(INSTANCES_MAP, host_id)

    # ── Pending hosts ──

    async def add_pending(self, pending_id: str, sid: str, data: dict) -> Optional[str]:
        """Add a pending host entry, deduplicating by API key.

        If a pending entry already exists for the same api_key, it is
        replaced by the new one (the latest connection supersedes).
        Returns the old pending_id that was replaced, or None.
        """
        r = redis_client()
        api_key = data.get("api_key", "")

        old_pending_id: Optional[str] = None
        if api_key:
            old_pending_id = await r.hget(PENDING_KEY_MAP, api_key)

        old_sid: Optional[str] = None
        if old_pending_id and old_pending_id != pending_id:
            old_raw = await r.hget(PENDING_MAP, old_pending_id)
            if old_raw:
                old_sid = json.loads(old_raw).get("sid")

        pipe = r.pipeline()
        if old_pending_id and old_pending_id != pending_id:
            pipe.hdel(PENDING_MAP, old_pending_id)
        if old_sid:
            pipe.hdel(PENDING_SID_MAP, old_sid)
        pipe.hset(PENDING_MAP, pending_id, json.dumps(data))
        pipe.hset(PENDING_SID_MAP, sid, pending_id)
        if api_key:
            pipe.hset(PENDING_KEY_MAP, api_key, pending_id)
        await pipe.execute()

        return (
            old_pending_id if old_pending_id and old_pending_id != pending_id else None
        )

    async def get_pending(self, pending_id: str) -> Optional[dict]:
        r = redis_client()
        raw = await r.hget(PENDING_MAP, pending_id)
        if raw is None:
            return None
        return json.loads(raw)

    async def get_pending_id_for_sid(self, sid: str) -> Optional[str]:
        r = redis_client()
        return await r.hget(PENDING_SID_MAP, sid)

    async def get_all_pending(self) -> List[dict]:
        r = redis_client()
        raw = await r.hgetall(PENDING_MAP)
        return [json.loads(v) for v in raw.values()]

    async def remove_pending(self, pending_id: str) -> Optional[dict]:
        r = redis_client()
        raw = await r.hget(PENDING_MAP, pending_id)
        if raw is None:
            return None
        data = json.loads(raw)
        sid = data.get("sid")
        api_key = data.get("api_key")
        pipe = r.pipeline()
        pipe.hdel(PENDING_MAP, pending_id)
        if sid:
            pipe.hdel(PENDING_SID_MAP, sid)
        if api_key:
            pipe.hdel(PENDING_KEY_MAP, api_key)
        await pipe.execute()
        return data

    async def remove_pending_by_sid(self, sid: str) -> Optional[str]:
        r = redis_client()
        pending_id = await r.hget(PENDING_SID_MAP, sid)
        if not pending_id:
            return None

        api_key: Optional[str] = None
        raw = await r.hget(PENDING_MAP, pending_id)
        if raw:
            api_key = json.loads(raw).get("api_key")

        pipe = r.pipeline()
        pipe.hdel(PENDING_SID_MAP, sid)
        pipe.hdel(PENDING_MAP, pending_id)
        if api_key:
            pipe.hdel(PENDING_KEY_MAP, api_key)
        await pipe.execute()
        return pending_id

    async def update_pending(self, pending_id: str, data: dict) -> None:
        r = redis_client()
        await r.hset(PENDING_MAP, pending_id, json.dumps(data))

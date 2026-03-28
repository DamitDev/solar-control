"""/hosts namespace - Solar hosts connect here to stream events.

Two-phase connection:
1. Host connects with auth={'api_key': '...'}
2. If the API key matches a registered host -> immediate activation
3. If the API key is unknown -> connection is accepted but held in
   "pending approval" state. Events are silently ignored until an
   admin approves the host from the WebUI.

All connection state is stored in Redis for multi-replica consistency.
"""

import asyncio
import logging
import uuid
from typing import Any
from datetime import datetime, timezone

from .server import sio
from app.database.hosts import host_db
from app.models import Host, HostStatus
from app.models.socketio import (
    HostHealthPayload,
    HostPendingPayload,
    HostStatusPayload,
    InstancesUpdatePayload,
    InstanceStatePayload,
    LogPayload,
    WSRegistration,
)
from app.redis_state import host_store

logger = logging.getLogger(__name__)


# ── Public helpers (async, backed by Redis) ───────────────────


async def get_host_instances(host_id: str) -> list[dict[str, Any]]:
    return await host_store.get_host_instances(host_id)


async def is_host_connected(host_id: str) -> bool:
    return await host_store.is_host_connected(host_id)


async def get_connected_host_ids() -> list[str]:
    return await host_store.get_connected_host_ids()


async def get_pending_hosts() -> list[dict[str, Any]]:
    return await host_store.get_all_pending()


async def get_pending_host(pending_id: str) -> dict[str, Any] | None:
    return await host_store.get_pending(pending_id)


def _api_key_preview(api_key: str) -> str:
    return api_key[:8] + "..." if len(api_key) > 8 else api_key


async def _emit_host_status(host: Host, *, connected: bool) -> None:
    """Emit a host_status event to WebUI using the typed payload model."""
    payload = HostStatusPayload.from_host(host, connected=connected)
    await sio.emit("host_status", payload.model_dump(), namespace="/webui")


async def approve_pending_host(pending_id: str, name: str, url: str) -> str | None:
    """Approve a pending host: create DB record, promote the socket connection.

    Returns the new host_id, or None if the pending_id was not found.
    """
    p = await host_store.remove_pending(pending_id)
    if not p:
        return None

    host_id = str(uuid.uuid4())
    gpu_type = p.get("gpu_type")
    roles = p.get("roles", [])
    host = Host(
        id=host_id,
        name=name,
        url=url,
        api_key=p["api_key"],
        status=HostStatus.ONLINE,
        gpu_type=gpu_type,
        roles=roles,
    )
    await host_db.add_host(host)

    sid: str = p["sid"]
    instances: list[dict[str, Any]] = p.get("instances", [])

    await host_store.register_host(sid, host_id)
    if instances:
        await host_store.set_host_instances(host_id, instances)

    await sio.emit(
        "registration_ack",
        {
            "host_id": host_id,
            "host_name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        to=sid,
        namespace="/hosts",
    )

    await sio.emit(
        "host_pending_removed", {"pending_id": pending_id}, namespace="/webui"
    )
    await _emit_host_status(host, connected=True)

    if instances:
        payload = InstancesUpdatePayload(host_id=host_id, instances=instances)
        await sio.emit("instances_update", payload.model_dump(), namespace="/webui")

    try:
        from app.gateway import gateway

        asyncio.create_task(gateway.refresh_model_registry())
    except Exception:
        pass

    logger.info("Pending host approved -> '%s' (%s)", name, host_id)
    return host_id


async def reject_pending_host(pending_id: str) -> bool:
    p = await host_store.remove_pending(pending_id)
    if not p:
        return False

    try:
        await sio.emit(
            "rejected",
            {"reason": "Host registration rejected by admin"},
            to=p["sid"],
            namespace="/hosts",
        )
        await sio.disconnect(p["sid"], namespace="/hosts")
    except Exception:
        pass

    await sio.emit(
        "host_pending_removed", {"pending_id": pending_id}, namespace="/webui"
    )
    logger.info("Pending host rejected (pending_id=%s)", pending_id)
    return True


# ── Socket.IO event handlers ─────────────────────────────────


@sio.on("connect", namespace="/hosts")
async def host_connect(
    sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None
):
    if not auth or "api_key" not in auth:
        logger.warning("Host %s rejected: no auth", sid)
        raise ConnectionRefusedError("Authentication required")

    api_key: str = auth["api_key"]
    host = await host_db.get_host_by_api_key(api_key)

    if host:
        await host_store.register_host(sid, host.id)
        await host_db.update_host_status(host.id, HostStatus.ONLINE)
        logger.info("Host '%s' (%s) connected [sid=%s]", host.name, host.id, sid)

        await sio.emit(
            "registration_ack",
            {
                "host_id": host.id,
                "host_name": host.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            to=sid,
            namespace="/hosts",
        )

        await _emit_host_status(host, connected=True)
    else:
        pending_id = str(uuid.uuid4())
        pending_data: dict[str, Any] = {
            "pending_id": pending_id,
            "sid": sid,
            "api_key": api_key,
            "host_name": auth.get("host_name", ""),
            "instances": [],
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
        await host_store.add_pending(pending_id, sid, pending_data)

        logger.info(
            "Host %s connected with unknown key -> pending (id=%s)", sid, pending_id
        )

        await sio.emit(
            "pending",
            {
                "pending_id": pending_id,
                "message": "Waiting for admin approval",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            to=sid,
            namespace="/hosts",
        )

        payload = HostPendingPayload(
            pending_id=pending_id,
            api_key_preview=_api_key_preview(api_key),
            host_name=pending_data["host_name"],
            connected_at=pending_data["connected_at"],
        )
        await sio.emit("host_pending", payload.model_dump(), namespace="/webui")


@sio.on("disconnect", namespace="/hosts")
async def host_disconnect(sid: str):
    host_id = await host_store.unregister_host_by_sid(sid)
    if host_id:
        await host_db.update_host_status(host_id, HostStatus.OFFLINE)

        host = await host_db.get_host(host_id)
        logger.info("Host '%s' (%s) disconnected", host.name if host else "?", host_id)

        if host:
            await _emit_host_status(host, connected=False)
        return

    pending_id = await host_store.remove_pending_by_sid(sid)
    if pending_id:
        logger.info("Pending host disconnected (pending_id=%s)", pending_id)
        await sio.emit(
            "host_pending_removed", {"pending_id": pending_id}, namespace="/webui"
        )


@sio.on("registration", namespace="/hosts")
async def host_registration(sid: str, data: dict[str, Any]):
    """Receive initial instance list, gpu_type, and roles from host."""
    reg = WSRegistration.model_validate(data)

    host_id = await host_store.get_host_id_for_sid(sid)
    if host_id:
        await host_store.set_host_instances(host_id, reg.instances)

        logger.info(
            "Registration from %s: gpu_type=%s, roles=%s, instances=%d",
            host_id,
            reg.gpu_type,
            reg.roles,
            len(reg.instances),
        )
        if reg.gpu_type or reg.roles:
            await host_db.update_host_registration(
                host_id,
                gpu_type=reg.gpu_type,
                roles=reg.roles or None,
            )

        payload = InstancesUpdatePayload(host_id=host_id, instances=reg.instances)
        await sio.emit("instances_update", payload.model_dump(), namespace="/webui")

        try:
            from app.gateway import gateway

            asyncio.create_task(gateway.refresh_model_registry())
        except Exception:
            pass
        return

    pending_id = await host_store.get_pending_id_for_sid(sid)
    if pending_id:
        p = await host_store.get_pending(pending_id)
        if p:
            p["host_name"] = reg.host_name or p.get("host_name", "")
            p["instances"] = reg.instances
            p["gpu_type"] = reg.gpu_type
            p["roles"] = reg.roles
            await host_store.update_pending(pending_id, p)

            payload = HostPendingPayload(
                pending_id=pending_id,
                api_key_preview=_api_key_preview(p["api_key"]),
                host_name=p["host_name"],
                instance_count=len(p["instances"]),
                connected_at=p["connected_at"],
            )
            await sio.emit("host_pending", payload.model_dump(), namespace="/webui")


@sio.on("instance_state", namespace="/hosts")
async def host_instance_state(sid: str, data: dict[str, Any]):
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    host = await host_db.get_host(host_id)
    payload = InstanceStatePayload(
        host_id=host_id,
        host_name=host.name if host else None,
        instance_id=data.get("instance_id"),
        timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        data=data.get("data", data),
    )
    await sio.emit("instance_state", payload.model_dump(), namespace="/webui")


@sio.on("log", namespace="/hosts")
async def host_log(sid: str, data: dict[str, Any]):
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    host = await host_db.get_host(host_id)
    payload = LogPayload(
        host_id=host_id,
        host_name=host.name if host else None,
        instance_id=data.get("instance_id"),
        timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        data=data.get("data", data),
    )
    await sio.emit("log", payload.model_dump(), namespace="/webui")


@sio.on("log_batch", namespace="/hosts")
async def host_log_batch(sid: str, data: dict[str, Any]):
    """Handle batched log entries from a host."""
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    entries: list[dict[str, Any]] = data.get("entries", [])
    if not entries:
        return

    host = await host_db.get_host(host_id)
    host_name = host.name if host else None

    for entry in entries:
        payload = LogPayload(
            host_id=host_id,
            host_name=host_name,
            instance_id=entry.get("instance_id"),
            timestamp=entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
            data={
                "seq": entry.get("seq"),
                "line": entry.get("line"),
                "level": entry.get("level", "info"),
            },
        )
        await sio.emit("log", payload.model_dump(), namespace="/webui")


@sio.on("instance_state_batch", namespace="/hosts")
async def host_instance_state_batch(sid: str, data: dict[str, Any]):
    """Handle batched instance state updates from a host."""
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    entries: list[dict[str, Any]] = data.get("entries", [])
    if not entries:
        return

    host = await host_db.get_host(host_id)
    host_name = host.name if host else None

    for entry in entries:
        payload = InstanceStatePayload(
            host_id=host_id,
            host_name=host_name,
            instance_id=entry.get("instance_id"),
            timestamp=entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
            data=entry.get("data", entry),
        )
        await sio.emit("instance_state", payload.model_dump(), namespace="/webui")


@sio.on("host_health", namespace="/hosts")
async def host_health(sid: str, data: dict[str, Any]):
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    health_data: dict[str, Any] = data.get("data", data)
    memory = health_data.get("memory")
    gpu_type = health_data.get("gpu_type")
    roles = health_data.get("roles")
    disk_total_gb = health_data.get("disk_total_gb")
    disk_used_gb = health_data.get("disk_used_gb")
    disk_available_gb = health_data.get("disk_available_gb")

    if memory:
        await host_db.update_host_memory(
            host_id,
            memory,
            gpu_type=gpu_type,
            disk_total_gb=disk_total_gb,
            disk_used_gb=disk_used_gb,
            disk_available_gb=disk_available_gb,
        )
    elif gpu_type:
        await host_db.update_host_gpu_type(host_id, gpu_type)

    if roles is not None:
        await host_db.update_host_roles(host_id, roles)

    host = await host_db.get_host(host_id)
    payload = HostHealthPayload(
        host_id=host_id,
        host_name=host.name if host else None,
        timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        data=health_data,
        memory=host.memory.model_dump() if host and host.memory else None,
        disk_total_gb=host.disk_total_gb if host else None,
        disk_used_gb=host.disk_used_gb if host else None,
        disk_available_gb=host.disk_available_gb if host else None,
        memory_available_gb=host.memory_available_gb if host else None,
    )
    await sio.emit("host_health", payload.model_dump(), namespace="/webui")


@sio.on("instances_update", namespace="/hosts")
async def host_instances_update(sid: str, data: dict[str, Any]):
    host_id = await host_store.get_host_id_for_sid(sid)
    if not host_id:
        return

    instances: list[dict[str, Any]] = data.get("data", {}).get(
        "instances", data.get("instances", [])
    )
    await host_store.set_host_instances(host_id, instances)

    payload = InstancesUpdatePayload(host_id=host_id, instances=instances)
    await sio.emit("instances_update", payload.model_dump(), namespace="/webui")

    try:
        from app.gateway import gateway

        asyncio.create_task(gateway.refresh_model_registry())
    except Exception:
        pass

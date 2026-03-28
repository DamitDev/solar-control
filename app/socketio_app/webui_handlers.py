"""/webui namespace - WebUI clients connect here to receive events.

Features:
- Authenticated via management API key
- Receives all host events (forwarded from /hosts namespace)
- Receives routing events (emitted by gateway)
- Can set filters for gateway_request events
- Receives initial status on connect
"""

import logging
from typing import Any

from .server import sio
from app.config import settings
from app.database.hosts import host_db
from app.models.socketio import HostStatusPayload, InstancesUpdatePayload
from app.socketio_app.host_handlers import (
    is_host_connected,
    get_pending_hosts,
    get_connected_host_ids,
    get_host_instances,
)

logger = logging.getLogger(__name__)


def _extract_key_from_environ(environ: dict[str, Any]) -> str | None:
    """Extract API key from ASGI scope headers (set by reverse proxy)."""
    for name, value in environ.get("headers", []):
        header = (
            name.decode("latin-1").lower() if isinstance(name, bytes) else name.lower()
        )
        val = value.decode("latin-1") if isinstance(value, bytes) else value
        if header == "x-api-key":
            return val
        if header == "authorization" and val.startswith("Bearer "):
            return val[7:]
    return None


@sio.on("connect", namespace="/webui")
async def webui_connect(
    sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None
):
    """Authenticate WebUI client and send initial state."""
    api_key = (auth or {}).get("api_key") or _extract_key_from_environ(environ)
    if api_key != settings.management_api_key:
        logger.warning("WebUI client %s rejected: bad auth", sid)
        raise ConnectionRefusedError("Invalid management API key")

    logger.info("WebUI client connected [sid=%s]", sid)

    hosts = await host_db.get_all_hosts()
    initial: list[dict[str, Any]] = []
    for h in hosts:
        payload = HostStatusPayload.from_host(
            h, connected=await is_host_connected(h.id)
        )
        initial.append(payload.model_dump())
    await sio.emit("initial_status", initial, to=sid, namespace="/webui")

    for hid in await get_connected_host_ids():
        instances = await get_host_instances(hid)
        if instances:
            payload = InstancesUpdatePayload(host_id=hid, instances=instances)
            await sio.emit(
                "instances_update", payload.model_dump(), to=sid, namespace="/webui"
            )

    pending = await get_pending_hosts()
    for p in pending:
        await sio.emit("host_pending", p, to=sid, namespace="/webui")


@sio.on("disconnect", namespace="/webui")
async def webui_disconnect(sid: str):
    logger.info("WebUI client disconnected [sid=%s]", sid)


@sio.on("set_filter", namespace="/webui")
async def webui_set_filter(sid: str, filter_config: dict[str, Any]):
    """Update the client's event filter."""
    async with sio.session(sid, namespace="/webui") as session:
        session["filter"] = filter_config

    await sio.emit(
        "filter_status",
        {"filter": filter_config},
        to=sid,
        namespace="/webui",
    )


async def broadcast_to_webui(event: str, data: dict[str, Any]) -> None:
    """Helper to emit events to all WebUI clients."""
    await sio.emit(event, data, namespace="/webui")


async def broadcast_gateway_request(summary_data: dict[str, Any]) -> None:
    """Broadcast a completed gateway request summary to WebUI clients."""
    await sio.emit("gateway_request", summary_data, namespace="/webui")

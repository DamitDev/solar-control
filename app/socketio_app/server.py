"""Socket.IO server with Redis adapter for cross-replica event broadcasting."""

import socketio

from app.config import settings

mgr = socketio.AsyncRedisManager(settings.redis_url)

sio = socketio.AsyncServer(
    async_mode="asgi",
    client_manager=mgr,
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_interval=25,
    ping_timeout=60,
)

sio_app = socketio.ASGIApp(sio, socketio_path="socket.io")

# Register namespace handlers (import triggers registration)
from . import host_handlers  # noqa: E402, F401
from . import webui_handlers  # noqa: E402, F401

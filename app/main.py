"""Solar Control - Stateless multi-replica coordinator with Socket.IO."""

import os
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version, PackageNotFoundError

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.auth import auth_middleware


def _get_version() -> str:
    try:
        return version("solar-control")
    except PackageNotFoundError:
        return "0.0.0-dev"


__version__ = _get_version()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn").setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("uvicorn.error").setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("uvicorn.access").setLevel(getattr(logging, log_level, logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import init_db, close_db
    from app.database.logs import gateway_logger
    from app.redis_state import init_redis, close_redis
    from app.gateway import gateway

    logger.info("Starting Solar Control v%s ...", __version__)

    await init_db(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    logger.info("PostgreSQL connected (pool_size=%d)", settings.db_pool_size)

    # Initialize Redis
    await init_redis(settings.redis_url)
    logger.info("Redis connected")

    # Start gateway logger (buffered writes)
    await gateway_logger.start()

    # Start gateway background tasks (registry refresh + health probes)
    await gateway.start_background_tasks()

    logger.info("Solar Control started successfully")

    yield

    logger.info("Shutting down Solar Control...")

    try:
        await gateway.stop_background_tasks()
    finally:
        await gateway.close()

    try:
        await gateway_logger.stop()
    except Exception as e:
        logger.error("Error stopping gateway logger: %s", e)

    await close_redis()
    await close_db()
    logger.info("Solar Control shut down")


# FastAPI app (handles REST routes)
app = FastAPI(
    title="Solar Control",
    description="Stateless coordinator for solar-host instances with multi-tenant OpenAI-compatible API gateway.",
    version=__version__,
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication middleware
app.middleware("http")(auth_middleware)

# Routes
from app.routes.openai import router as openai_router  # noqa: E402
from app.routes.management import router as management_router  # noqa: E402

app.include_router(openai_router)
app.include_router(management_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "solar-control", "version": __version__}


@app.get("/ready")
async def readiness_check():
    """Readiness probe - checks DB and Redis connectivity."""
    try:
        from sqlalchemy import text
        from app.database.connection import get_session_factory
        from app.redis_state.connection import redis_client

        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        r = redis_client()
        await r.ping()
        return {"status": "ready"}
    except Exception as e:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503, content={"status": "not_ready", "error": str(e)}
        )


@app.get("/")
async def root():
    return {
        "service": "solar-control",
        "version": __version__,
        "description": "Stateless multi-replica coordinator with multi-tenant OpenAI gateway",
        "endpoints": {
            "openai": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/classify",
                "/v1/embeddings",
                "/v1/rerank",
            ],
            "management": [
                "/api/hosts",
                "/api/endpoints",
                "/api/gateway/stats",
                "/api/gateway/requests",
                "/api/models/availability",
                "/api/models/distribute",
                "/api/resources",
                "/api/instances/migrate",
            ],
            "realtime": [
                "Socket.IO /hosts (host connections)",
                "Socket.IO /webui (WebUI connections)",
            ],
        },
    }


# Mount Socket.IO on top of FastAPI
from app.socketio_app import sio  # noqa: E402

sio_asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:sio_asgi_app", host=settings.host, port=settings.port, reload=True
    )

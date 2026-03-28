"""SQLAlchemy async engine and session factory for PostgreSQL."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_db() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory


async def init_db(database_url: str) -> AsyncEngine:
    global _engine, _session_factory

    sa_url = database_url
    if sa_url.startswith("postgresql://"):
        sa_url = sa_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(sa_url, pool_size=10, max_overflow=5)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def close_db() -> None:
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None

"""Redis connection pool for shared state."""

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None


def redis_client() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _client


async def init_redis(redis_url: str) -> aioredis.Redis:
    global _client
    _client = aioredis.from_url(redis_url, decode_responses=True)
    await _client.ping()
    return _client


async def close_redis() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None

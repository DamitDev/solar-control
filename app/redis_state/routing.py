"""Routing state in Redis: active request counts, host weights, round-robin.

All operations are atomic to ensure consistency across replicas.
"""

from .connection import redis_client

ACTIVE_PREFIX = "solar:active:"
WEIGHT_PREFIX = "solar:weight:"
RR_PREFIX = "solar:rr:"

# TTL for active counters - auto-cleanup if a replica crashes mid-request
ACTIVE_TTL_S = 300


class RoutingStore:
    """Atomic routing state shared across all solar-control replicas."""

    # --- Active request counts per instance ---

    async def increment_active(self, host_id: str, instance_id: str) -> int:
        r = redis_client()
        key = f"{ACTIVE_PREFIX}{host_id}:{instance_id}"
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, ACTIVE_TTL_S)
        val, _ = await pipe.execute()
        return val

    async def decrement_active(self, host_id: str, instance_id: str) -> int:
        r = redis_client()
        key = f"{ACTIVE_PREFIX}{host_id}:{instance_id}"
        val = await r.decr(key)
        if val <= 0:
            await r.delete(key)
            return 0
        return val

    async def get_active(self, host_id: str, instance_id: str) -> int:
        r = redis_client()
        val = await r.get(f"{ACTIVE_PREFIX}{host_id}:{instance_id}")
        return int(val) if val else 0

    # --- Host active weight (sum of model sizes in B) ---

    async def add_weight(self, host_id: str, weight: float) -> float:
        r = redis_client()
        key = f"{WEIGHT_PREFIX}{host_id}"
        pipe = r.pipeline()
        pipe.incrbyfloat(key, weight)
        pipe.expire(key, ACTIVE_TTL_S)
        val, _ = await pipe.execute()
        return val

    async def remove_weight(self, host_id: str, weight: float) -> float:
        r = redis_client()
        key = f"{WEIGHT_PREFIX}{host_id}"
        val = await r.incrbyfloat(key, -weight)
        if val <= 0:
            await r.delete(key)
            return 0.0
        return val

    async def get_weight(self, host_id: str) -> float:
        r = redis_client()
        val = await r.get(f"{WEIGHT_PREFIX}{host_id}")
        return float(val) if val else 0.0

    # --- Host active count (total requests on a host) ---

    async def increment_host_active(self, host_id: str) -> int:
        r = redis_client()
        key = f"{ACTIVE_PREFIX}host:{host_id}"
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, ACTIVE_TTL_S)
        val, _ = await pipe.execute()
        return val

    async def decrement_host_active(self, host_id: str) -> int:
        r = redis_client()
        key = f"{ACTIVE_PREFIX}host:{host_id}"
        val = await r.decr(key)
        if val <= 0:
            await r.delete(key)
            return 0
        return val

    async def get_host_active(self, host_id: str) -> int:
        r = redis_client()
        val = await r.get(f"{ACTIVE_PREFIX}host:{host_id}")
        return int(val) if val else 0

    # --- Round-robin per model ---

    RR_TTL_S = 3600

    async def next_rr_index(self, model: str) -> int:
        """Get and increment the round-robin index for a model."""
        r = redis_client()
        key = f"{RR_PREFIX}{model}"
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.RR_TTL_S)
        val, _ = await pipe.execute()
        return val

"""Model registry stored in Redis.

Maps model aliases to lists of instance entries, shared across all replicas.
"""

import json
from typing import Any

from .connection import redis_client

REGISTRY_KEY = "solar:registry"


class RegistryStore:
    """Read/write model-to-instances mapping in Redis."""

    async def set_registry(
        self, model_to_hosts: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Replace the entire registry atomically."""
        r = redis_client()
        pipe = r.pipeline()
        pipe.delete(REGISTRY_KEY)
        if model_to_hosts:
            mapping = {
                model: json.dumps(instances)
                for model, instances in model_to_hosts.items()
            }
            pipe.hset(REGISTRY_KEY, mapping=mapping)
        await pipe.execute()

    async def get_registry(self) -> dict[str, list[dict[str, Any]]]:
        """Get the full registry."""
        r = redis_client()
        raw = await r.hgetall(REGISTRY_KEY)
        return {model: json.loads(instances) for model, instances in raw.items()}

    async def get_instances_for_model(self, model: str) -> list[dict[str, Any]]:
        """Get instance list for a specific model alias."""
        r = redis_client()
        raw = await r.hget(REGISTRY_KEY, model)
        if raw is None:
            return []
        return json.loads(raw)

    async def get_all_model_names(self) -> list[str]:
        r = redis_client()
        return list(await r.hkeys(REGISTRY_KEY))

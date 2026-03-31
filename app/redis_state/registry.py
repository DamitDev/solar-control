"""Model registry stored in Redis.

Maps model aliases to lists of RegistryEntry instances, shared across all replicas.
"""

import json

from app.models.gateway import RegistryEntry

from .connection import redis_client

REGISTRY_KEY = "solar:registry"


class RegistryStore:
    """Read/write model-to-instances mapping in Redis."""

    async def set_registry(
        self, model_to_instances: dict[str, list[RegistryEntry]]
    ) -> None:
        """Replace the entire registry atomically."""
        r = redis_client()
        pipe = r.pipeline()
        pipe.delete(REGISTRY_KEY)
        if model_to_instances:
            mapping = {
                model: json.dumps([e.model_dump() for e in entries])
                for model, entries in model_to_instances.items()
            }
            pipe.hset(REGISTRY_KEY, mapping=mapping)
        await pipe.execute()

    async def get_registry(self) -> dict[str, list[RegistryEntry]]:
        """Get the full registry."""
        r = redis_client()
        raw = await r.hgetall(REGISTRY_KEY)
        return {
            model: [RegistryEntry.model_validate(e) for e in json.loads(data)]
            for model, data in raw.items()
        }

    async def get_instances_for_model(self, model: str) -> list[RegistryEntry]:
        """Get instance list for a specific model alias."""
        r = redis_client()
        raw = await r.hget(REGISTRY_KEY, model)
        if raw is None:
            return []
        return [RegistryEntry.model_validate(e) for e in json.loads(raw)]

    async def get_all_model_names(self) -> list[str]:
        r = redis_client()
        return list(await r.hkeys(REGISTRY_KEY))

"""Gateway-specific models for the model registry and routing."""

from typing import Any, ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


class RegistryEntry(BaseModel):
    """A routable instance in the model registry.

    Each entry represents one running model instance on one host,
    carrying everything the gateway needs to route a request to it.
    """

    model_config = ConfigDict(protected_namespaces=())

    host_id: str
    instance_id: str
    url: str
    api_key: str
    model_alias: str
    supported_endpoints: list[str] = Field(default_factory=list)
    backend_type: str = "llamacpp"

    DEFAULT_ENDPOINTS: ClassVar[list[str]] = [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
    ]

    @classmethod
    def from_ws_instance(
        cls,
        host_id: str,
        host_url: str,
        host_api_key: str,
        instance: dict[str, Any],
    ) -> "RegistryEntry | None":
        """Build from a WS-pushed or Redis-cached instance dict.

        These have a flat structure with ``alias``, ``port``,
        ``supported_endpoints``, ``backend_type`` at the top level.
        The API key comes from the host, not the instance.
        """
        port = instance.get("port")
        if not port:
            return None
        parsed = urlparse(host_url)
        instance_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
        return cls(
            host_id=host_id,
            instance_id=instance["id"],
            url=instance_url,
            api_key=host_api_key,
            model_alias=instance.get("alias", "unknown"),
            supported_endpoints=instance.get(
                "supported_endpoints", cls.DEFAULT_ENDPOINTS
            ),
            backend_type=instance.get("backend_type", "llamacpp"),
        )

    @classmethod
    def from_http_instance(
        cls,
        host_id: str,
        host_url: str,
        instance: dict[str, Any],
    ) -> "RegistryEntry | None":
        """Build from an HTTP-polled instance dict (solar-host REST API).

        These have a nested ``config`` dict containing ``alias``,
        ``api_key``, and ``backend_type``.
        """
        port = instance.get("port")
        if not port:
            return None
        config = instance.get("config", {})
        parsed = urlparse(host_url)
        instance_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
        return cls(
            host_id=host_id,
            instance_id=instance["id"],
            url=instance_url,
            api_key=config.get("api_key", ""),
            model_alias=config.get("alias", "unknown"),
            supported_endpoints=instance.get(
                "supported_endpoints", cls.DEFAULT_ENDPOINTS
            ),
            backend_type=config.get("backend_type", "llamacpp"),
        )

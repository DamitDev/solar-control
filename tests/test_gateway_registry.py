from unittest.mock import AsyncMock, patch

import pytest

from app.gateway import OpenAIGateway
from app.models import Host, HostStatus, RegistryEntry


class _Response:
    def __init__(self, status: int, payload=None):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload


class _RequestContext:
    def __init__(self, response: _Response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    closed = False

    def __init__(self, response: _Response):
        self._response = response

    def get(self, *args, **kwargs):
        return _RequestContext(self._response)


@pytest.fixture
def host():
    return Host(
        id="host-1",
        name="Test Host",
        url="http://test-host:8000",
        api_key="host-api-key",
        status=HostStatus.ONLINE,
    )


@pytest.mark.anyio
async def test_refresh_recovers_connected_host_with_empty_instance_cache(host):
    instances = [
        {
            "id": "inst-1",
            "status": "running",
            "port": 3500,
            "supported_endpoints": ["/v1/chat/completions", "/v1/models"],
            "config": {
                "alias": "model-a",
                "api_key": "instance-api-key",
                "backend_type": "llamacpp",
            },
        }
    ]

    gateway = OpenAIGateway()
    gateway.session = _Session(_Response(200, instances))

    with (
        patch("app.gateway.host_db.get_all_hosts", AsyncMock(return_value=[host])),
        patch("app.gateway.host_db.update_host_status", AsyncMock()),
        patch(
            "app.socketio_app.host_handlers.is_host_connected",
            AsyncMock(return_value=True),
        ),
        patch(
            "app.socketio_app.host_handlers.get_host_instances",
            AsyncMock(return_value=[]),
        ),
        patch(
            "app.gateway.host_store.get_disconnect_time", AsyncMock(return_value=None)
        ),
        patch("app.gateway.host_store.set_host_instances", AsyncMock()) as set_cache,
        patch("app.gateway.registry_store.set_registry", AsyncMock()) as set_registry,
    ):
        await gateway.refresh_model_registry()

    set_cache.assert_awaited_once_with(
        "host-1",
        [
            {
                "id": "inst-1",
                "alias": "model-a",
                "status": "running",
                "port": 3500,
                "supported_endpoints": ["/v1/chat/completions", "/v1/models"],
                "backend_type": "llamacpp",
                "api_key": "instance-api-key",
            }
        ],
    )

    registry = set_registry.await_args.args[0]
    assert list(registry) == ["model-a"]
    assert registry["model-a"][0].instance_id == "inst-1"
    assert registry["model-a"][0].api_key == "instance-api-key"


@pytest.mark.anyio
async def test_refresh_keeps_previous_registry_when_polling_fails(host):
    previous = {
        "model-a": [
            RegistryEntry(
                host_id="host-1",
                instance_id="inst-1",
                url="http://test-host:3500",
                api_key="key",
                model_alias="model-a",
            )
        ]
    }

    gateway = OpenAIGateway()
    gateway.session = _Session(_Response(503))

    with (
        patch("app.gateway.host_db.get_all_hosts", AsyncMock(return_value=[host])),
        patch("app.gateway.host_db.update_host_status", AsyncMock()),
        patch(
            "app.socketio_app.host_handlers.is_host_connected",
            AsyncMock(return_value=False),
        ),
        patch(
            "app.gateway.host_store.get_disconnect_time", AsyncMock(return_value=None)
        ),
        patch("app.gateway.host_store.get_host_instances", AsyncMock(return_value=[])),
        patch(
            "app.gateway.registry_store.get_registry", AsyncMock(return_value=previous)
        ),
        patch("app.gateway.registry_store.set_registry", AsyncMock()) as set_registry,
    ):
        await gateway.refresh_model_registry()

    set_registry.assert_not_awaited()

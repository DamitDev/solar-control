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


class _URLSession:
    closed = False

    def __init__(self, responses: dict[str, _Response]):
        self._responses = responses

    def get(self, url, *args, **kwargs):
        return _RequestContext(self._responses[url])


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


@pytest.mark.anyio
async def test_http_registry_preserves_llamacpp_context_size(host):
    instances = [
        {
            "id": "inst-1",
            "status": "running",
            "port": 3500,
            "supported_endpoints": ["/v1/chat/completions", "/v1/models"],
            "config": {
                "alias": "qwen3.6:35b",
                "api_key": "instance-api-key",
                "backend_type": "llamacpp",
                "ctx_size": 40960,
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
            AsyncMock(return_value=False),
        ),
        patch(
            "app.gateway.host_store.get_disconnect_time",
            AsyncMock(return_value=None),
        ),
        patch("app.gateway.host_store.get_host_instances", AsyncMock(return_value=[])),
        patch("app.gateway.host_store.set_host_instances", AsyncMock()),
        patch("app.gateway.registry_store.set_registry", AsyncMock()) as set_registry,
    ):
        await gateway.refresh_model_registry()

    registry = set_registry.await_args.args[0]
    assert registry["qwen3.6:35b"][0].context_size == 40960


@pytest.mark.anyio
async def test_models_response_overrides_llamacpp_context_metadata(host):
    registry_entry = RegistryEntry(
        host_id="host-1",
        instance_id="inst-1",
        url="http://test-host:3500",
        api_key="instance-api-key",
        model_alias="qwen3.6:35b",
        context_size=40960,
    )
    upstream_models = {
        "models": [
            {
                "name": "qwen3.6:35b",
                "model": "qwen3.6:35b",
                "details": {"format": "gguf"},
                "capabilities": ["completion"],
            }
        ],
        "data": [
            {
                "id": "qwen3.6:35b",
                "object": "model",
                "owned_by": "llamacpp",
                "meta": {"n_ctx_train": 262144, "n_params": 34660610688},
            }
        ],
    }

    gateway = OpenAIGateway()
    gateway.session = _URLSession(
        {"http://test-host:3500/v1/models": _Response(200, upstream_models)}
    )

    with patch(
        "app.gateway.registry_store.get_registry",
        AsyncMock(return_value={"qwen3.6:35b": [registry_entry]}),
    ):
        result = await gateway.get_available_models()

    assert result["data"][0]["meta"]["n_ctx_train"] == 40960
    assert result["data"][0]["meta"]["ctx_size"] == 40960
    assert result["data"][0]["capabilities"] == ["completion"]
    assert result["models"][0]["details"]["context_length"] == 40960


@pytest.mark.anyio
async def test_models_response_fetches_missing_context_size(host):
    registry_entry = RegistryEntry(
        host_id="host-1",
        instance_id="inst-1",
        url="http://test-host:3500",
        api_key="instance-api-key",
        model_alias="qwen3.6:35b",
    )
    upstream_models = {
        "models": [],
        "data": [
            {
                "id": "qwen3.6:35b",
                "object": "model",
                "owned_by": "llamacpp",
                "meta": {"n_ctx_train": 262144},
            }
        ],
    }
    instance_details = {
        "id": "inst-1",
        "config": {"backend_type": "llamacpp", "ctx_size": 40960},
    }

    gateway = OpenAIGateway()
    gateway.session = _URLSession(
        {
            "http://test-host:3500/v1/models": _Response(200, upstream_models),
            "http://test-host:8000/instances/inst-1": _Response(200, instance_details),
        }
    )

    with (
        patch(
            "app.gateway.registry_store.get_registry",
            AsyncMock(return_value={"qwen3.6:35b": [registry_entry]}),
        ),
        patch("app.gateway.host_db.get_host", AsyncMock(return_value=host)),
    ):
        result = await gateway.get_available_models()

    assert result["data"][0]["meta"]["n_ctx_train"] == 40960

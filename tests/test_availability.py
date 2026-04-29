import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.models import Host, HostStatus
from app.routes.management.models import (
    _fetch_host_models,
    get_model_availability,
    ModelAvailabilityResponse,
)


@pytest.fixture
def mock_host():
    return Host(
        id="host-1",
        name="Test Host",
        url="http://test-host:8000",
        api_key="test-key",
        status=HostStatus.ONLINE,
    )


@pytest.fixture
def mock_host_2():
    return Host(
        id="host-2",
        name="Test Host 2",
        url="http://test-host-2:8000",
        api_key="test-key-2",
        status=HostStatus.ONLINE,
    )


@pytest.mark.anyio
async def test_fetch_host_models_success(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = [
            {"name": "model-a", "size_bytes": 1000, "path": "/path/a"},
            {"name": "model-b", "size_bytes": 2000, "path": "/path/b"},
        ]
        mock_get.return_value.__aenter__.return_value = mock_resp

        models = await _fetch_host_models(mock_host)
        assert len(models) == 2
        assert models[0]["name"] == "model-a"
        assert models[1]["name"] == "model-b"


@pytest.mark.anyio
async def test_fetch_host_models_non200(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_get.return_value.__aenter__.return_value = mock_resp

        models = await _fetch_host_models(mock_host)
        assert models == []


@pytest.mark.anyio
async def test_fetch_host_models_connection_error(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        import aiohttp

        mock_get.side_effect = aiohttp.ClientConnectionError("Refused")

        models = await _fetch_host_models(mock_host)
        assert models == []


@pytest.mark.anyio
async def test_fetch_host_models_timeout(mock_host):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = asyncio.TimeoutError()

        models = await _fetch_host_models(mock_host)
        assert models == []


@pytest.mark.anyio
async def test_get_model_availability_aggregates(mock_host, mock_host_2):
    hosts = [mock_host, mock_host_2]

    # host-1 has model-a, model-b
    # host-2 has model-a, model-c
    mock_results = [
        [
            {"name": "model-a", "size_bytes": 1000, "path": "/path/a1"},
            {"name": "model-b", "size_bytes": 2000, "path": "/path/b1"},
        ],
        [
            {"name": "model-a", "size_bytes": 1000, "path": "/path/a2"},
            {"name": "model-c", "size_bytes": 3000, "path": "/path/c2"},
        ],
    ]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch("app.routes.management.models._fetch_host_models") as mock_fetch,
    ):
        mock_fetch.side_effect = mock_results

        resp = await get_model_availability()
        assert isinstance(resp, ModelAvailabilityResponse)
        assert "model-a" in resp.models
        assert len(resp.models["model-a"]) == 2
        assert resp.models["model-a"][0].host_id == "host-1"
        assert resp.models["model-a"][1].host_id == "host-2"

        assert "model-b" in resp.models
        assert len(resp.models["model-b"]) == 1
        assert resp.models["model-b"][0].host_id == "host-1"

        assert "model-c" in resp.models
        assert len(resp.models["model-c"]) == 1
        assert resp.models["model-c"][0].host_id == "host-2"


@pytest.mark.anyio
async def test_get_model_availability_filter_found(mock_host, mock_host_2):
    hosts = [mock_host, mock_host_2]
    mock_results = [
        [{"name": "model-a", "size_bytes": 1000, "path": "/path/a1"}],
        [{"name": "model-b", "size_bytes": 2000, "path": "/path/b2"}],
    ]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch("app.routes.management.models._fetch_host_models") as mock_fetch,
    ):
        mock_fetch.side_effect = mock_results

        resp = await get_model_availability(model_name="model-a")
        assert "model-a" in resp.models
        assert "model-b" not in resp.models
        assert len(resp.models) == 1


@pytest.mark.anyio
async def test_get_model_availability_filter_not_found(mock_host):
    hosts = [mock_host]
    mock_results = [[{"name": "model-a", "size_bytes": 1000, "path": "/path/a1"}]]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch("app.routes.management.models._fetch_host_models") as mock_fetch,
    ):
        mock_fetch.side_effect = mock_results

        resp = await get_model_availability(model_name="non-existent")
        assert resp.models == {}


@pytest.mark.anyio
async def test_get_model_availability_no_hosts():
    with patch("app.database.hosts.host_db.get_all_hosts", return_value=[]):
        resp = await get_model_availability()
        assert resp.models == {}


@pytest.mark.anyio
async def test_get_model_availability_one_unreachable(mock_host, mock_host_2):
    hosts = [mock_host, mock_host_2]
    # host-1 succeeds, host-2 fails (returns [])
    mock_results = [[{"name": "model-a", "size_bytes": 1000, "path": "/path/a1"}], []]

    with (
        patch("app.database.hosts.host_db.get_all_hosts", return_value=hosts),
        patch("app.routes.management.models._fetch_host_models") as mock_fetch,
    ):
        mock_fetch.side_effect = mock_results

        resp = await get_model_availability()
        assert "model-a" in resp.models
        assert len(resp.models["model-a"]) == 1
        assert resp.models["model-a"][0].host_id == "host-1"
